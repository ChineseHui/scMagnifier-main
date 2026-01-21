#!/usr/bin/env python3
"""
scMagnifier/spatial_perturb.py
功能：
- 无依赖celloracle的空间转录组扰动分析
- 基于STAGATE降维，结合GRN系数矩阵实现基因扰动
- 输出扰动后表达矩阵、聚类结果、UMAP/spatial图等
"""
import warnings
warnings.filterwarnings("ignore")

# ========================================================
# TensorFlow（必须最先导入）
# ========================================================
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

# ========================================================
# 常规依赖
# ========================================================
import os
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
from typing import Optional

import STAGATE
from STAGATE.utils import Cal_Spatial_Net, Stats_Spatial_Net


# ========================================================
# GRN扰动传播工具函数
# ========================================================
def propagate_matrix(deltaX, coef_mat, pert_indices, n_iters=3):
    """
    GRN系数矩阵驱动的扰动传播计算
    
    Parameters
    ----------
    deltaX : np.ndarray
        初始扰动矩阵（cells × genes）
    coef_mat : np.ndarray
        聚类特异性GRN系数矩阵
    pert_indices : list
        被扰动基因的索引列表
    n_iters : int, optional
        传播迭代次数，默认3
    
    Returns
    -------
    np.ndarray
        传播后的扰动矩阵
    """
    v = deltaX.copy()
    for _ in range(n_iters):
        v_next = v @ coef_mat
        v_next[:, pert_indices] = deltaX[:, pert_indices]
        v = v_next
    return v

# ========================================================
# 单multiplier扰动分析函数（内部调用）
# ========================================================
def _run_for_one_multiplier(
    input: str,
    grn_npz: str,
    gene_list: str,
    multiplier: float,
    alpha: float,
    spot_size: Optional[int],
    resolution: float,
    stagate_epochs: int,
    k_cutoff: int,
    plot_umap: bool,
    save_h5ad: bool,
    outdir: str
):
    """内部函数：处理单个multiplier的扰动分析"""
    # 格式化multiplier字符串（用于目录命名）
    mult_str = str(multiplier).replace(".", "p").replace("-", "neg")
    mult_outdir = os.path.join(outdir, mult_str)
    os.makedirs(mult_outdir, exist_ok=True)

    print(f"\n[INFO] === Running multiplier={multiplier} ===")

    # 1. 加载输入数据
    adata = sc.read_h5ad(input)
    coef_npz = np.load(grn_npz, allow_pickle=True)

    # 2. 提取原始count矩阵（确保为稠密数组）
    if "raw_count" in adata.layers:
        X_raw = adata.layers["raw_count"]
        if not isinstance(X_raw, np.ndarray):
            X_raw = X_raw.toarray()
    else:
        X_raw = adata.X
        if not isinstance(X_raw, np.ndarray):
            X_raw = X_raw.toarray()
        print("[WARN] raw_count layer not found, using dense adata.X instead")

    if X_raw.ndim != 2:
        raise ValueError(f"X_raw must be 2D (cells × genes), got shape {X_raw.shape}")

    # 3. 保存原始表达矩阵
    pd.DataFrame(X_raw, columns=adata.var_names, index=adata.obs_names).to_csv(
        os.path.join(mult_outdir, "original_matrix.csv")
    )
    print(f"[SAVED] original_matrix.csv to {mult_outdir}")

    # 4. 准备基础信息（固定使用leiden聚类）
    gene_names = list(adata.var_names)
    cluster_labels = adata.obs["leiden"].astype(str)  # 固定cluster_key为leiden
    clusters = sorted(cluster_labels.unique())
    old_umap = adata.obsm["X_umap"].copy() if "X_umap" in adata.obsm else None

    # 5. 读取待扰动基因列表
    with open(gene_list) as f:
        genes_to_perturb = [g.strip() for g in f if g.strip()]

    # 6. 逐基因执行扰动分析
    for gene in genes_to_perturb:
        if gene not in gene_names:
            print(f"[WARN] Gene {gene} not found in adata, skip")
            continue

        print(f"[RUNNING] Perturbing gene={gene}")
        pert_idx = gene_names.index(gene)
        X_new_total = X_raw.copy()

        # 7. 按聚类执行GRN扰动传播
        for cid in clusters:
            if cid not in coef_npz:
                continue
            mask = cluster_labels == cid
            if mask.sum() == 0:
                continue

            coef_mat = coef_npz[cid]
            deltaX = np.zeros_like(X_raw[mask])
            deltaX[:, pert_idx] = X_raw[mask][:, pert_idx] * multiplier
            deltaX_prop = propagate_matrix(deltaX, coef_mat, [pert_idx])
            X_new_total[mask] += deltaX_prop

        # 确保表达量非负
        X_new_total = np.maximum(X_new_total, 0)

        # 8. 保存扰动后表达矩阵
        pert_matrix_path = os.path.join(mult_outdir, f"perturbed_matrix_{gene}_{multiplier}.csv")
        pd.DataFrame(
            X_new_total, columns=gene_names, index=adata.obs_names
        ).to_csv(pert_matrix_path)
        print(f"[SAVED] {pert_matrix_path}")

        # 9. STAGATE下游分析
        ad = adata.copy()
        ad.X = X_new_total.copy()
        sc.pp.log1p(ad)

        # STAGATE空间网络构建与训练
        Cal_Spatial_Net(ad, k_cutoff=k_cutoff, model="KNN")
        Stats_Spatial_Net(ad)
        STAGATE.train_STAGATE(ad, alpha=alpha, n_epochs=stagate_epochs)

        # UMAP与Leiden聚类
        sc.pp.neighbors(ad, use_rep="STAGATE")
        sc.tl.umap(ad)
        sc.tl.leiden(ad, resolution=resolution, key_added="leiden_perturbed")

        # 10. 保存聚类结果
        cluster_csv_path = os.path.join(mult_outdir, f"cluster_{gene}_{multiplier}.csv")
        ad.obs[["leiden_perturbed"]].to_csv(cluster_csv_path)
        print(f"[SAVED] {cluster_csv_path}")

        # 11. 保存扰动后h5ad（受控）
        if save_h5ad:
            h5ad_path = os.path.join(mult_outdir, f"{gene}_mult{mult_str}_perturbed.h5ad")
            ad.write_h5ad(h5ad_path)
            print(f"[SAVED] {h5ad_path}")

        # 12. 绘图（UMAP/Spatial）
        if plot_umap:
            # 新UMAP图
            fig_umap_new = sc.pl.umap(ad, color="leiden_perturbed", show=False, return_fig=True)
            fig_umap_new.savefig(
                os.path.join(mult_outdir, f"umap_new_{gene}_{multiplier}.png"),
                dpi=600,
                bbox_inches="tight"
            )
            plt.close(fig_umap_new)

            # Spatial图（动态添加spot_size参数）
            spatial_kwargs = {"color": "leiden_perturbed", "show": False}
            if spot_size is not None:
                spatial_kwargs["spot_size"] = spot_size
            sc.pl.spatial(ad,** spatial_kwargs)
            plt.savefig(
                os.path.join(mult_outdir, f"spatial_{gene}_{multiplier}.png"),
                dpi=600,
                bbox_inches="tight"
            )
            plt.close()

            # 旧UMAP背景的新聚类图
            if old_umap is not None:
                ad.obsm["X_umap"] = old_umap.copy()
                fig_umap_old = sc.pl.umap(ad, color="leiden_perturbed", show=False, return_fig=True)
                fig_umap_old.savefig(
                    os.path.join(mult_outdir, f"umap_old_on_new_{gene}_{multiplier}.png"),
                    dpi=600,
                    bbox_inches="tight"
                )
                plt.close(fig_umap_old)

        print(f"[DONE] Perturbation for gene {gene} completed")

# ========================================================
# 核心对外函数：spatial_perturb
# ========================================================
def spatial_perturb(
    # 核心输入参数（默认值按要求对齐）
    input: str = os.path.join("preprocessed_result", "preprocessed.h5ad"),
    grn_npz: str = os.path.join("GRN", "celloracle_grn_coef.npz"),
    gene_list: str = os.path.join("GRN", "TF_HVG_intersect_genes.txt"),
    multiplier: str = "0.1,-0.1",  
    alpha: float = 0,
    spot_size: Optional[int] = None,
    resolution: float = 0.3,
    stagate_epochs: int = 300,
    k_cutoff: int = 6,
    # 输出控制参数
    plot_umap: bool = True,
    save_h5ad: bool = False,
    outdir: str = "perturb_results"
) -> None:
    """
    空间转录组基因扰动分析函数（基于STAGATE+GRN系数矩阵）
    
    Parameters
    ----------
    input : str, optional
        输入预处理后的h5ad文件路径，默认：preprocessed_result/preprocessed.h5ad（对齐spatial_preprocess输出）
    grn_npz : str, optional
        GRN系数矩阵npz文件路径，默认：GRN/celloracle_grn_coef.npz（对齐spatial_preperturb输出）
    gene_list : str, optional
        待扰动基因列表文件路径，默认：GRN/TF_HVG_intersect_genes.txt（对齐GRN函数输出）
    multiplier : str, optional
        扰动倍数（逗号分隔字符串），默认："0.1,-0.1"
    alpha : float, optional
        STAGATE训练的alpha参数，默认0（对齐spatial_preprocess）
    spot_size : int, optional
        Spatial图的spot大小，传入数值则添加该参数，不传则不添加（对齐spatial_preprocess）
    n_hvg : int, optional
        高可变基因数量（原代码未使用，仅对齐参数规范），默认2000
    min_counts : int, optional
        基因最低表达量过滤阈值（原代码未使用，仅对齐参数规范），默认1
    resolution : float, optional
        Leiden聚类分辨率，默认0.3（对齐spatial_preprocess）
    seed : int, optional
        随机种子（原代码未使用，仅对齐参数规范），默认42
    stagate_epochs : int, optional
        STAGATE训练轮数，默认300（对齐spatial_preprocess）
    k_cutoff : int, optional
        STAGATE构建空间网络的KNN邻居数，默认6（对齐spatial_preprocess）
    plot_umap : bool, optional
        是否绘制UMAP/Spatial图，默认True
    save_h5ad : bool, optional
        是否保存扰动后h5ad文件，默认False
    outdir : str, optional
        扰动结果输出目录，默认"perturb_results"
    
    Raises
    ------
    FileNotFoundError
        输入文件（input/grn_npz/gene_list）不存在时抛出
    ValueError
        输入矩阵维度错误、multiplier格式错误时抛出
    """
    # 1. 输入校验
    for file_path, desc in zip([input, grn_npz, gene_list], ["input h5ad", "GRN npz", "gene list"]):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"{desc} file not found: {file_path}")

    # 2. 创建输出根目录
    os.makedirs(outdir, exist_ok=True)
    print(f"[INFO] Output root directory: {os.path.abspath(outdir)}")

    # 3. 解析multiplier列表
    try:
        multipliers = [float(m.strip()) for m in multiplier.split(",")]
    except ValueError:
        raise ValueError(f"Invalid multiplier format: {multiplier}\nHint: Use comma-separated floats (e.g., '0.1,-0.1')")

    # 4. 逐multiplier执行扰动分析
    for m in multipliers:
        _run_for_one_multiplier(
            input=input,
            grn_npz=grn_npz,
            gene_list=gene_list,
            multiplier=m,
            alpha=alpha,
            spot_size=spot_size,
            resolution=resolution,
            stagate_epochs=stagate_epochs,
            k_cutoff=k_cutoff,
            plot_umap=plot_umap,
            save_h5ad=save_h5ad,
            outdir=outdir
        )

    print("\n[INFO] All multipliers completed! All results saved to:", os.path.abspath(outdir))
