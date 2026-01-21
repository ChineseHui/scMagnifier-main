#!/usr/bin/env python3
"""
multi_preprocess.py

Unified preprocessing pipeline for batch correction (Harmony/Scanorama/scVI)
Supports multi-batch data processing, with strict consistency to original per-method logic.

Core pipeline (shared by all methods):
 1) filter genes by min counts
 2) normalize per cell (TOTAL UMI) -- no log
 3) select HVG on non-log normalized data
 4) subset to HVG
 5) re-normalize per cell (TOTAL UMI)
 6) save un-logged copy (adata.raw + layers['raw_count'])

Method-specific steps:
- harmony: PCA → Harmony batch correction → neighbors (X_pca_harmony) → UMAP → clustering
- scanorama: log1p+scale → Scanorama integration → neighbors (X_scanorama) → UMAP → clustering
- scVI: scVI on raw_count → latent z (X_scVI) → neighbors → UMAP → clustering

Usage:
  from scMagnifier import multi_preprocess
  # 极简使用（默认method=harmony，batch_key=batch）
  multi_preprocess(input_path="data.h5ad")
  # 自定义算法
  multi_preprocess(method="scvi", input_path="data.h5ad", batch_key="sample_batch")
"""
import warnings
warnings.filterwarnings("ignore")

import os
import re
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
plt.switch_backend('Agg')

# Method-specific imports (lazy import to avoid missing dependency errors)
try:
    import harmonypy
except ImportError:
    harmonypy = None

try:
    from scanorama import integrate_scanpy
except ImportError:
    integrate_scanpy = None

try:
    import torch
    from scvi.model import SCVI
except ImportError:
    torch = None
    SCVI = None

# --------------------------
# scVI特有函数
# --------------------------
def run_one_scvi_returning_adata(adata, seed, args):
    """
    Train scVI on a copy of adata (expects adata.layers['raw_count'] exists)
    Returns an AnnData with obsm['X_scVI'] set to the learned latent representation.
    This function follows your earlier signature / settings.
    """
    if "raw_count" not in adata.layers:
        raise ValueError("raw_count layer not found in adata.layers")

    ad = adata.copy()
    # set seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    # register anndata for scvi using the provided batch_key
    SCVI.setup_anndata(ad, layer="raw_count", batch_key=args.batch_key)
    model = SCVI(ad, n_latent=args.n_pcs, gene_likelihood="nb")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Train: honor args.max_epochs
    model.train(
        max_epochs=args.max_epochs,
        accelerator=device,
        enable_progress_bar=False
    )

    latent = model.get_latent_representation()
    ad.obsm["X_scVI"] = latent

    return ad

# --------------------------
# 核心整合函数：multi_preprocess（method默认harmony）
# --------------------------
def multi_preprocess(
    input_path: str,  # 必填：输入h5ad文件路径
    method: str = "harmony",  # 可选：批次校正算法（默认harmony，支持scanorama/scvi）
    batch_key: str = "batch",  # 可选：adata.obs中表示批次的列名（默认"batch"）
    outdir: str = "preprocessed_result",  # 可选：输出目录（默认自动创建）
    output_name: str = "preprocessed.h5ad",  # 可选：输出文件名称
    n_hvg: int = 2000,  # 可选：HVG数量（默认2000）
    min_counts: int = 1,  # 可选：基因最小计数（默认1）
    n_pcs: int = 20,  # 可选：PCA维度（Harmony/scVI）/无实际作用（Scanorama）（默认20）
    n_neighbors: int = 10,  # 可选：近邻数（默认10）
    resolution: float = 0.75,  # 可选：聚类分辨率（默认0.75）
    seed: int = 42,  # 可选：随机种子（默认42）
    cluster_method: str = "leiden",  # 可选：聚类方法（默认leiden，支持louvain）
    label_key: str = "assigned_cluster",  # 自定义标签名称列，默认"assigned_cluster"
    max_epochs: int = 100,  # 可选：scVI训练最大轮数（仅scVI生效，默认100）
) -> None:
    """
    多批次数据预处理函数（整合Harmony/Scanorama/scVI）
    严格保持每个算法的原始处理逻辑，默认使用Harmony进行批次校正。

    参数说明：
    ----------
    input_path : str
        输入h5ad文件路径（必填）
    method : str, optional
        批次校正算法，可选值：harmony/scanorama/scvi，默认值："harmony"
    batch_key : str, optional
        adata.obs中表示批次的列名，默认值："batch"
    outdir : str, optional
        输出目录（默认自动创建），默认值："preprocessed_result"
    output_name : str, optional
        输出h5ad文件名，默认值："preprocessed.h5ad"
    n_hvg : int, optional
        高变基因数量，默认值：2000
    min_counts : int, optional
        基因过滤的最小计数，默认值：1
    n_pcs : int, optional
        - Harmony: PCA的维度
        - scVI: 潜在空间维度
        - Scanorama: 无实际作用（保留以兼容参数）
        默认值：20
    n_neighbors : int, optional
        构建近邻图的邻居数，默认值：10
    resolution : float, optional
        聚类分辨率（Leiden/Louvain），默认值：0.75
    seed : int, optional
        随机种子，默认值：42
    cluster_method : str, optional
        聚类方法，可选leiden/louvain，默认值：leiden
    label_key : str, optional
        adata.obs中表示细胞类型/标签的列名（用于绘图），默认值："assigned_cluster"
    max_epochs : int, optional
        scVI训练的最大轮数，仅scVI生效，默认值：100

    异常：
    ----------
    ValueError: method不合法/缺少依赖/batch_key不存在等
    """
    # 预处理参数
    method = method.lower()
    cluster_method = cluster_method.lower()
    valid_methods = ["harmony", "scanorama", "scvi"]
    if method not in valid_methods:
        raise ValueError(f"method必须是{valid_methods}之一，当前输入：{method}")
    
    # 检查依赖
    if method == "harmony" and harmonypy is None:
        raise ImportError("请安装harmonypy: pip install harmonypy")
    if method == "scanorama" and integrate_scanpy is None:
        raise ImportError("请安装scanorama: pip install scanorama")
    if method == "scvi" and (torch is None or SCVI is None):
        raise ImportError("请安装scvi-tools: pip install scvi-tools")
    
    # 固定随机种子
    np.random.seed(seed)
    if method == "scvi" and torch is not None:
        torch.manual_seed(seed)

    # 创建输出目录
    os.makedirs(outdir, exist_ok=True)

    # --------------------------
    # 步骤1-6：所有方法共享的预处理
    # --------------------------
    print("Loading data from:", input_path)
    adata = sc.read_h5ad(input_path)
    print("Original data shape:", adata.shape)

    # Step 1: 过滤基因
    print(f"Filtering genes with min_counts >= {min_counts} ...")
    sc.pp.filter_genes(adata, min_counts=min_counts)
    print("After gene filtering shape:", adata.shape)

    # Step 2: 归一化（总UMI，无log）
    print("Normalizing per cell (total counts -> target_sum=1e4) ...")
    sc.pp.normalize_total(adata, target_sum=1e4)

    # Step 3: 选择HVG（非log归一化数据）
    print(f"Selecting top {n_hvg} highly variable genes (on non-log data) ...")
    sc.pp.highly_variable_genes(
        adata, n_top_genes=n_hvg, flavor='cell_ranger', 
        subset=False, inplace=True
    )
    hvgs = adata.var.highly_variable.values
    n_hvgs = hvgs.sum()
    print(f"Number of HVG selected: {n_hvgs}")

    # Step 4: 子集到HVG
    print("Subsetting AnnData to HVGs ...")
    adata = adata[:, adata.var['highly_variable']].copy()
    print("After subsetting shape:", adata.shape)

    # Step 5: 重归一化
    print("Re-normalizing per cell after gene filtering ...")
    sc.pp.normalize_total(adata, target_sum=1e4)

    # Step 6: 保存未log的副本
    print("Saving un-logged copy to adata.raw and adata.layers['raw_count'] ...")
    adata.raw = adata.copy()
    try:
        raw_X = adata.raw.X.toarray() if hasattr(adata.raw.X, "toarray") else adata.raw.X.copy()
    except Exception:
        raw_X = adata.raw.X.copy()
    adata.layers["raw_count"] = raw_X

    # --------------------------
    # 方法特有分支
    # --------------------------
    use_rep = None  # 用于neighbors的representation
    adata_processed = adata.copy()

    if method == "harmony":
        # ====== Harmony分支======
        # Step 7: log1p + scale
        print("Applying log1p and scaling ...")
        print("Before log1p: min=", adata_processed.X.min(), "max=", adata_processed.X.max(), "mean=", adata_processed.X.mean())
        sc.pp.log1p(adata_processed)
        sc.pp.scale(adata_processed, max_value=10)

        # Step 8: PCA
        print(f"Running PCA (n_comps={n_pcs}) ...")
        sc.tl.pca(adata_processed, svd_solver='arpack', n_comps=n_pcs)

        # Harmony批次校正
        print(f"Performing Harmony batch correction using key '{batch_key}' ...")
        if batch_key not in adata_processed.obs.columns:
            raise ValueError(f"ERROR: batch_key='{batch_key}' was not found in adata.obs!")
        
        sc.external.pp.harmony_integrate(
            adata_processed,
            key=batch_key,
            basis='X_pca',
            adjusted_basis='X_pca_harmony',
            random_state=seed
        )
        print("Harmony integration complete. Corrected PCs saved in adata.obsm['X_pca_harmony'].")
        use_rep = "X_pca_harmony"

    elif method == "scanorama":
        # ====== Scanorama分支======
        # Step 7: log1p + scale
        print("Applying log1p and scaling ...")
        print("Before log1p: min=", adata_processed.X.min(), "max=", adata_processed.X.max(), "mean=", adata_processed.X.mean())
        sc.pp.log1p(adata_processed)
        sc.pp.scale(adata_processed, max_value=10)

        # Scanorama批次校正
        print(f"Performing Scanorama integration using key '{batch_key}' ...")
        if batch_key not in adata_processed.obs.columns:
            raise ValueError(f"ERROR: batch_key='{batch_key}' was not found in adata.obs!")
        
        # 拆分批次
        batches = adata_processed.obs[batch_key].unique().tolist()
        print("Found batches:", batches)
        adatas = [adata_processed[adata_processed.obs[batch_key] == b].copy() for b in batches]

        # 整合
        try:
            integrate_scanpy(adatas)
        except Exception as e:
            raise RuntimeError("Scanorama integration failed: " + str(e))
        
        # 确保每个adata有X_scanorama
        for ad in adatas:
            if 'X_scanorama' not in ad.obsm:
                ad.obsm['X_scanorama'] = ad.X.copy()
        
        # 合并回一个adata
        merged = adatas[0].concatenate(*adatas[1:], batch_key="batch_sc", index_unique=None)
        if 'X_scanorama' not in merged.obsm:
            try:
                X_scanorama_list = [ad.obsm['X_scanorama'] for ad in adatas]
                merged.obsm['X_scanorama'] = np.vstack(X_scanorama_list)
            except Exception:
                merged.obsm['X_scanorama'] = merged.X.copy()
        
        adata_processed = merged
        print("Scanorama integration complete. Corrected embedding saved in adata.obsm['X_scanorama'].")
        use_rep = "X_scanorama"

    elif method == "scvi":
        # ====== scVI分支======
        # 注意：scVI分支跳过log1p+scale，直接用raw_count训练
        print("Running scVI on raw_count to obtain latent representation (will be used instead of PCA) ...")
        if batch_key not in adata_processed.obs.columns:
            raise ValueError(f"ERROR: batch_key='{batch_key}' not found in adata.obs!")
        
        ad_scvi = run_one_scvi_returning_adata(
            adata_processed, seed=seed, 
            args=type('Args', (), {'batch_key': batch_key, 'n_pcs': n_pcs, 'max_epochs': max_epochs})()
        )
        adata_processed = ad_scvi
        print("scVI training complete. latent stored in adata.obsm['X_scVI'].")
        use_rep = "X_scVI"

    # --------------------------
    # 步骤9-13：所有方法共享的后处理（聚类、绘图、保存）
    # --------------------------
    # Step 9: 构建近邻图
    print(f"Computing neighbors (n_neighbors={n_neighbors}) using '{use_rep}' ...")
    sc.pp.neighbors(adata_processed, use_rep=use_rep, n_neighbors=n_neighbors, random_state=seed)

    # Step 10: UMAP
    print(f"Computing UMAP based on {method}-corrected embedding ...")
    sc.tl.umap(adata_processed, random_state=seed)

    # Step 11: 聚类
    if cluster_method == "leiden":
        print(f"Running Leiden clustering (resolution={resolution}) ...")
        sc.tl.leiden(adata_processed, resolution=resolution, key_added="leiden", random_state=seed)
        cluster_key = "leiden"
    elif cluster_method == "louvain":
        print(f"Running Louvain clustering (resolution={resolution}) ...")
        sc.tl.louvain(adata_processed, resolution=resolution, key_added="louvain", random_state=seed)
        cluster_key = "louvain"
    else:
        raise ValueError("cluster_method must be either 'leiden' or 'louvain'")

    # Step 12: 绘制UMAP（聚类结果）
    print(f"Plotting UMAP colored by {cluster_key} clusters ...")
    fig1 = sc.pl.umap(
        adata_processed, color=cluster_key, size=8, alpha=0.8,
        show=False, return_fig=True
    )
    current_title = fig1.axes[0].get_title()
    fig1.axes[0].set_title(current_title, fontsize=10)
    legend = fig1.axes[0].get_legend()
    if legend:
        for text in legend.get_texts():
            text.set_fontsize(10)
    umap_png = os.path.join(outdir, f"umap_{cluster_key}_{method}.png")
    fig1.savefig(umap_png, bbox_inches='tight')
    plt.close(fig1)
    print("UMAP figure saved to:", umap_png)

    # Step 13: 绘制UMAP（label_key）
    print(f"Trying to plot UMAP using label_key='{label_key}' ...")
    if label_key in adata_processed.obs.columns:
        print(f"Plotting UMAP colored by {label_key} ...")
        fig2 = sc.pl.umap(
            adata_processed, color=label_key, size=8, alpha=0.8,
            show=False, return_fig=True
        )
        current_title = fig2.axes[0].get_title()
        fig2.axes[0].set_title(current_title, fontsize=10)
        legend = fig2.axes[0].get_legend()
        if legend:
            for text in legend.get_texts():
                text.set_fontsize(10)
        umap_pred_png = os.path.join(outdir, f"umap_{label_key}_{method}.png")
        fig2.savefig(umap_pred_png, bbox_inches='tight')
        plt.close(fig2)
        print("UMAP (label_key) figure saved to:", umap_pred_png)
    else:
        print(f"Warning: label_key='{label_key}' not found in adata.obs. Skipping label UMAP plot.")

    # Step 14: 保存结果
    out_h5ad = os.path.join(outdir, output_name)
    adata_processed.write(out_h5ad)
    print("Preprocessed AnnData saved to:", out_h5ad)

    print("Preprocessing finished. Outputs saved in", outdir)
    print("Note: adata.raw and adata.layers['raw_count'] contain un-logged data for downstream use.")
