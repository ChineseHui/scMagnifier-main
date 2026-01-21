#!/usr/bin/env python3
"""
preprocess.py

Preprocessing pipeline for single-cell RNA-seq data (adapted for CellOracle):
 1) filter genes by min counts
 2) normalize per cell (TOTAL UMI) -- no log
 3) select HVG on the (non-log) normalized data
 4) subset to HVG
 5) re-normalize per cell (TOTAL UMI)
 6) save an un-logged copy (adata.raw and layers['raw_count'])
 7) log1p transform and scale
 8) PCA -> neighbors -> UMAP -> clustering (Leiden or Louvain)
All random operations are fixed by a seed.

Package usage:
    from your_package_name import preprocess
    
    # 基础使用（只指定输入路径，其他用默认值）
    preprocess(input_path="/mnt/disk1/hzh/sc/benchmark/single/EBUS_10.h5ad")
    
    # 自定义参数使用（包括自定义label_key）
    preprocess(
        input_path="/path/to/your/data.h5ad",
        outdir="my_output",
        output_name="my_preprocessed.h5ad",
        n_hvg=3000,
        cluster_method="louvain",
        label_key="Cell_type" 
    )
"""
import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import scanpy as sc
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path  # 新增：更优雅的路径处理


plt.switch_backend('Agg')


def preprocess(
    input_path: str,  # 必填：输入h5ad文件路径
    outdir: str = "preprocessed_result",  # 可选：输出目录（默认自动创建）
    output_name: str = "preprocessed.h5ad",  # 可选：输出文件名称
    n_hvg: int = 2000,  # 可选：HVG数量（默认2000）
    min_counts: int = 1,  # 可选：基因最小计数（默认1）
    n_pcs: int = 20,  # 可选：PCA维度（默认20）
    n_neighbors: int = 10,  # 可选：近邻数（默认10）
    resolution: float = 0.75,  # 可选：聚类分辨率（默认0.75）
    seed: int = 42,  # 可选：随机种子（默认42）
    cluster_method: str = "leiden",  # 可选：聚类方法（默认leiden，支持louvain）
    label_key: str = "Cell_subtype"  #自定义标签名称列，默认"Cell_subtype"
) -> None:
    """
    单细胞数据预处理核心函数（适配CellOracle）
    
    参数说明：
    ----------
    input_path : str
        输入h5ad文件的绝对/相对路径（必填，无默认值）
    outdir : str, optional
        输出目录名称（默认：preprocessed_result），会自动创建
    output_name : str, optional
        预处理后h5ad文件的名称（默认：preprocessed.h5ad）
    n_hvg : int, optional
        选择的高可变基因数量（默认：2000）
    min_counts : int, optional
        基因保留的最小计数阈值（默认：1）
    n_pcs : int, optional
        PCA降维后的维度数（默认：20）
    n_neighbors : int, optional
        计算近邻时的邻居数（默认：10）
    resolution : float, optional
        聚类分辨率（值越大，聚类数越多，默认：0.75）
    seed : int, optional
        随机种子（保证结果可复现，默认：42）
    cluster_method : str, optional
        聚类方法，可选"leiden"或"louvain"（默认：leiden）
    label_key : str, optional
        adata.obs中细胞类型/亚型的列名（默认："Cell_subtype"），支持自定义
    
    返回值：
    ----------
    None
        结果会保存到指定输出目录，包含预处理后的h5ad文件和UMAP图
    """
    # 固定随机种子（保证结果可复现）
    np.random.seed(seed)
    
    # 路径处理：转换为Path对象，自动兼容Windows/Linux
    input_path = Path(input_path)
    outdir = Path(outdir)
    
    # 校验输入文件是否存在
    if not input_path.exists():
        raise FileNotFoundError(f"输入文件不存在：{input_path.absolute()}")
    
    # 自动创建输出目录（如果不存在）
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"Loading outdir：{outdir.absolute()}")

    # 加载数据
    print(f"Loading data from: {input_path.absolute()}")
    adata = sc.read_h5ad(input_path)
    print(f"Original data shape: {adata.shape}")

    # Step 1: 过滤基因（最小计数）
    print(f"Filtering genes with min_counts >= {min_counts} ...")
    sc.pp.filter_genes(adata, min_counts=min_counts)
    print(f"After gene filtering shape: {adata.shape}")

    # Step 2: 细胞水平归一化（非log）
    print("Normalizing per cell (total counts -> target_sum=1e4) ...")
    sc.pp.normalize_total(adata, target_sum=1e4)

    # Step 3: 选择HVG（非log数据）
    print(f"Selecting top {n_hvg} highly variable genes (on non-log data) ...")
    sc.pp.highly_variable_genes(
        adata, 
        n_top_genes=n_hvg, 
        flavor='cell_ranger', 
        subset=False, 
        inplace=True
    )
    hvgs = adata.var.highly_variable.values
    n_hvgs = hvgs.sum()
    print(f"Number of HVG selected: {n_hvgs}")

    # Step 4: 子集到HVG
    print("Subsetting AnnData to HVGs ...")
    adata = adata[:, adata.var['highly_variable']].copy()
    print(f"After subsetting shape: {adata.shape}")

    # Step 5: 重新归一化
    print("Re-normalizing per cell after gene filtering ...")
    sc.pp.normalize_total(adata, target_sum=1e4)

    # Step 6: 保存未log的原始数据
    print("Saving un-logged copy to adata.raw and adata.layers['raw_count'] ...")
    adata.raw = adata.copy()
    try:
        raw_X = adata.raw.X.toarray() if hasattr(adata.raw.X, "toarray") else adata.raw.X.copy()
    except Exception:
        raw_X = adata.raw.X.copy()
    adata.layers["raw_count"] = raw_X

    # Step 7: log1p + 标准化
    print("Applying log1p and scaling ...")
    print(f"Before log1p: min={adata.X.min()}, max={adata.X.max()}, mean={adata.X.mean()}")
    sc.pp.log1p(adata)
    sc.pp.scale(adata, max_value=10)

    # Step 8: PCA
    print(f"Running PCA (n_comps={n_pcs}) ...")
    sc.tl.pca(adata, svd_solver='arpack', n_comps=n_pcs)

    # Step 9: 计算近邻
    print(f"Computing neighbors (n_neighbors={n_neighbors}, n_pcs={n_pcs}) ...")
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs, random_state=seed)

    # Step 10: UMAP
    print("Computing UMAP ...")
    sc.tl.umap(adata, random_state=seed)

    # Step 11: 聚类
    cluster_method = cluster_method.lower()
    if cluster_method == "leiden":
        print(f"Running Leiden clustering (resolution={resolution}) ...")
        sc.tl.leiden(adata, resolution=resolution, key_added="leiden", random_state=seed)
        cluster_key = "leiden"
    elif cluster_method == "louvain":
        print(f"Running Louvain clustering (resolution={resolution}) ...")
        sc.tl.louvain(adata, resolution=resolution, key_added="louvain", random_state=seed)
        cluster_key = "louvain"
    else:
        raise ValueError("cluster_method must be either 'leiden' or 'louvain'")

    # -----------------------------
    # Step 12: Plot UMAP (cluster)
    # Use Leiden-style palette for all plots
    # -----------------------------
    print(f"Plotting UMAP colored by {cluster_key} clusters ...")
    fig1 = sc.pl.umap(adata, color=cluster_key, size=8, alpha=0.8, show=False, return_fig=True)
    
    # 设置标题和大小
    current_title = fig1.axes[0].get_title()
    fig1.axes[0].set_title(current_title,fontsize=10)

    # 设置图例大小
    legend = fig1.axes[0].get_legend()
    if legend:
        for text in legend.get_texts():
            text.set_fontsize(10)

    umap_png = os.path.join(outdir, f"umap_{cluster_key}.png")
    fig1.savefig(umap_png, bbox_inches='tight')
    plt.close(fig1)
    print(f"UMAP ({cluster_key}) figure saved to:", umap_png)

    # Step 12b: Plot UMAP
    if label_key in adata.obs.columns:
        print(f"Plotting UMAP colored by {label_key} labels  ...")
        fig2 = sc.pl.umap(adata, color=label_key,size=8, alpha=0.8, show=False, return_fig=True)
        
        # 设置标题和大小
        current_title = fig2.axes[0].get_title()
        fig2.axes[0].set_title(current_title,fontsize=10)
        
        # 设置图例大小
        legend = fig2.axes[0].get_legend()
        if legend:
            for text in legend.get_texts():
                text.set_fontsize(10)
                
        # 文件名包含自定义的label_key，避免覆盖
        umap_pred_png = outdir / f"umap_{label_key}.png"
        fig2.savefig(umap_pred_png, bbox_inches='tight')
        plt.close(fig2)
        print(f"UMAP ({label_key}) figure saved to: {umap_pred_png.absolute()}")
    else:
        # 提示信息包含自定义的label_key
        print(f"Warning: '{label_key}' column not found in adata.obs, skipping UMAP by {label_key}.")

    # Step 13: Save preprocessed AnnData
    out_h5ad = outdir / output_name
    adata.write_h5ad(out_h5ad)
    print(f"Preprocessed AnnData saved to: {out_h5ad.absolute()}")

    print("="*50)
    print("Preprocessing finished! All outputs saved in:", outdir.absolute())
    print("Note: adata.raw and adata.layers['raw_count'] contain un-logged data for downstream use.")