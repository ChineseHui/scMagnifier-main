#!/usr/bin/env python3
"""
scMagnifier/spatial_preprocess.py
STAGATE 空间转录组预处理函数
核心功能：读取h5ad文件，过滤空白spot，STAGATE降维，生成UMAP/spatial图
"""
import warnings
warnings.filterwarnings("ignore")

# ========================================================
#  TensorFlow（必须最先导入，STAGATE依赖）
# ========================================================
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

# ========================================================
#  常规依赖
# ========================================================
import os
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
from typing import Optional

import STAGATE
from STAGATE.utils import Cal_Spatial_Net, Stats_Spatial_Net
plt.switch_backend('Agg')

# ========================================================
#  核心预处理函数
# ========================================================
def spatial_preprocess(
    input_path: str,
    outdir: str = "preprocessed_result",
    output_name: str = "preprocessed.h5ad",
    alpha: float = 0,
    spot_size: Optional[int] = None,
    n_hvg: int = 2000,
    min_counts: int = 1,
    resolution: float = 0.3,
    seed: int = 42,
    stagate_epochs: int = 300,
    k_cutoff: int = 6
) -> sc.AnnData:
    """
    STAGATE 空间转录组数据预处理函数（适配Visium数据，过滤空白spot）

    Parameters
    ----------
    input_path : str
        必填：输入h5ad文件的绝对/相对路径
    outdir : str, optional
        可选：输出结果目录，默认 "preprocessed_result"（自动创建）
    output_name : str, optional
        可选：输出h5ad文件名称，默认 "preprocessed.h5ad"
    alpha : float, optional
        可选：STAGATE训练的alpha参数，默认 0（可自定义修改）
    spot_size : int, optional
        可选：spatial图的spot大小，传入数值则添加该参数，不传则不添加
    n_hvg : int, optional
        可选：高可变基因数量，默认 2000
    min_counts : int, optional
        可选：基因最低表达量过滤阈值，默认 1
    resolution : float, optional
        可选：Leiden聚类分辨率，默认 0.3
    seed : int, optional
        可选：随机种子，保证结果可复现，默认 42
    stagate_epochs : int, optional
        可选：STAGATE训练轮数，默认 300
    k_cutoff : int, optional
        可选：STAGATE构建空间网络的KNN邻居数，默认 10

    Returns
    -------
    sc.AnnData
        预处理后的AnnData对象（包含STAGATE降维、UMAP、Leiden聚类结果）

    Raises
    ------
    FileNotFoundError
        输入h5ad文件路径不存在时抛出
    """
    # 1. 输入校验
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"输入文件不存在：{input_path}")
    
    # 2. 初始化配置
    np.random.seed(seed)
    os.makedirs(outdir, exist_ok=True)

    # 3. 读取h5ad文件
    adata = sc.read_h5ad(input_path)
    adata.var_names_make_unique()
    print(f"Original data shape: {adata.shape}")

    # 4. 过滤空白spot
    if "in_tissue" in adata.obs.columns:
        adata = adata[adata.obs["in_tissue"] == 1].copy()
        print(f"Data shape after filtering blank spots: {adata.shape}")

    # 5. 表达矩阵预处理
    # 5.1 基因表达量过滤
    sc.pp.filter_genes(adata, min_counts=min_counts)
    # 5.2 高可变基因筛选
    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=n_hvg,
        flavor="cell_ranger",
        subset=False
    )
    adata = adata[:, adata.var["highly_variable"]].copy()
    # 5.3 归一化+对数转换
    sc.pp.normalize_total(adata, target_sum=1e4)
    adata.raw = adata.copy()
    # 保存原始count矩阵到layers
    raw_X = adata.raw.X.toarray() if hasattr(adata.raw.X, "toarray") else adata.raw.X.copy()
    adata.layers["raw_count"] = raw_X
    sc.pp.log1p(adata)

    # 6. STAGATE 空间网络构建+训练
    print("Starting STAGATE ...")
    Cal_Spatial_Net(adata, k_cutoff=k_cutoff, model="KNN")
    Stats_Spatial_Net(adata)
    STAGATE.train_STAGATE(adata, alpha=alpha, n_epochs=stagate_epochs)

    # 7. 下游分析（UMAP+Leiden聚类）
    sc.pp.neighbors(adata, use_rep="STAGATE")
    sc.tl.umap(adata)
    sc.tl.leiden(adata, resolution=resolution, key_added="leiden")

    # 8. 绘图
    # 8.1 UMAP图
    fig_umap = sc.pl.umap(adata, color="leiden", show=False, return_fig=True)
    fig_umap.savefig(
        os.path.join(outdir, "umap_leiden.png"),
        dpi=600,
        bbox_inches="tight"
    )
    plt.close(fig_umap)

    # 8.2 Spatial图
    spatial_kwargs = {"color": "leiden", "show": False}
    if spot_size is not None:
        spatial_kwargs["spot_size"] = spot_size  # 仅传入spot_size时添加该参数
    sc.pl.spatial(adata,** spatial_kwargs)
    plt.savefig(
        os.path.join(outdir, "spatial_leiden.png"),
        dpi=600,
        bbox_inches="tight"
    )
    plt.close()

    # 9. 保存结果
    adata.write(os.path.join(outdir, output_name))
    print(f"Preprocessing completed! Results saved to: {os.path.join(outdir, output_name)}")
