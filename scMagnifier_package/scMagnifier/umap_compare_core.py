#!/usr/bin/env python3
"""
umap_compare.py

核心功能：
 - 在同一坐标轴上绘制左右两个UMAP（原始UMAP + 自定义rpcUMAP）
 - 按CSV定义的簇着色，支持高亮指定簇并绘制跨UMAP的连线
 - 自适应偏移确保两个UMAP帧无重叠，帧大小严格覆盖所有点
 - 适配包的默认路径规范，开箱即用

Package usage:
    from scMagnifier import umap_compare
    
    # 基础使用（默认路径）
    umap_compare.plot_umap_compare(selection=["1"])
    
    # 自定义参数
    umap_compare.plot_umap_compare(
        h5ad_path="/path/to/adata.h5ad",
        cluster_csv_path="/path/to/merged_clusters.csv",
        selection=["1", "3"],
        out_file="/path/to/umap_compare.png",
        point_size=6
    )
"""
import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib
matplotlib.use("Agg")  # 非交互式后端，适合批量绘图
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle



# ----------------------------------------------------
# 工具函数（内部使用，不对外暴露）
# ----------------------------------------------------
def _find_umap_in_obsm(adata, candidates):
    """
    从adata.obsm中查找UMAP嵌入矩阵
    内部工具函数，不对外暴露
    
    Parameters
    ----------
    adata : anndata.AnnData
        包含UMAP嵌入的AnnData对象
    candidates : list[str]
        UMAP键的候选列表
    
    Returns
    -------
    str | None
        找到的UMAP键
    np.ndarray | None
        UMAP嵌入矩阵（仅取前两列）
    """
    for k in candidates:
        if k in adata.obsm:
            arr = np.asarray(adata.obsm[k])
            if arr.ndim == 2 and arr.shape[1] >= 2:
                return k, arr[:, :2]
    return None, None

def _make_palette(categories):
    """
    为簇标签生成颜色映射
    内部工具函数，不对外暴露
    
    Parameters
    ----------
    categories : list
        簇标签列表
    
    Returns
    -------
    dict
        {簇标签: 颜色}的映射字典
    """
    pal = sns.color_palette(None, len(categories))
    return {cat: pal[i] for i, cat in enumerate(categories)}

# ----------------------------------------------------
# 核心对外函数（包的暴露接口）
# ----------------------------------------------------
def umap_compare(
    selection: list,
    h5ad_path: str = os.path.join("consensus_result", "adata_with_rpcumap.h5ad"),
    cluster_csv_path: str = None,
    out_file: str = None,
    outdir: str = "umap_compare",
    point_size: int = 4,
    alpha: float = 0.8,
    line_alpha: float = 0.4
) -> None:
    """
    在同一坐标轴绘制原始UMAP和自定义rpcUMAP的对比图，支持高亮指定簇
    
    Parameters
    ----------
    selection : list[str | int]
        需要高亮的簇ID列表（如["1", "3"]或[1, 3]）
    h5ad_path : str, optional
        输入h5ad文件路径，默认："consensus_result/adata_with_rpcumap.h5ad"
    cluster_csv_path : str | None, optional
        簇定义CSV文件路径（格式：cell_id,cluster），默认None时自动使用outdir下的merged_clusters.csv
    out_file : str | None, optional
        输出图片路径，默认None时自动保存为outdir/umap_compare.png
    outdir : str, optional
        输出目录（仅在cluster_csv_path或out_file为None时生效），默认："merged_result"
    point_size : int, optional
        散点大小，默认：4
    alpha : float, optional
        散点透明度，默认：0.8
    line_alpha : float, optional
        高亮簇连线透明度，默认：0.4
    
    Raises
    ------
    ValueError
        找不到所需的UMAP嵌入矩阵时抛出
    FileNotFoundError
        h5ad或CSV文件不存在时抛出
    
    Examples
    --------
    >>> # 默认路径使用
    >>> plot_umap_compare(selection=["1"])
    
    >>> # 自定义路径
    >>> plot_umap_compare(
    ...     selection=[1, 3],
    ...     h5ad_path="/path/to/adata.h5ad",
    ...     cluster_csv_path="/path/to/merged_clusters.csv",
    ...     out_file="/path/to/umap_compare.png"
    ... )
    """
    # --------------------
    # 参数默认值补全（适配包的目录规范）
    # --------------------
    # 补全cluster_csv_path默认值
    if cluster_csv_path is None:
        cluster_csv_path = os.path.join("merged_result", "merged_clusters.csv")
    # 补全out_file默认值
    if out_file is None:
        os.makedirs(outdir, exist_ok=True)
        out_file = os.path.join(outdir, "umap_compare.png")
    
    # --------------------
    # 文件存在性校验
    # --------------------
    if not os.path.exists(h5ad_path):
        raise FileNotFoundError(
            f"h5ad file not found: {h5ad_path}\n"
            f"Hint: Check if the consensus function has been executed"
        )
    if not os.path.exists(cluster_csv_path):
        raise FileNotFoundError(
            f"Cluster definition CSV file not found: {cluster_csv_path}\n"
            f"Hint: Check if the merge function has been executed"
        )
    
    # --------------------
    # 加载数据
    # --------------------
    print(f"[INFO] Loading h5ad file: {h5ad_path}")
    adata = sc.read_h5ad(h5ad_path)
    
    print(f"[INFO] Loading cluster definition CSV: {cluster_csv_path}")
    new_clust_df = pd.read_csv(cluster_csv_path, dtype=str)
    
    # --------------------
    # 查找UMAP嵌入矩阵
    # --------------------
    _, left_umap = _find_umap_in_obsm(adata, ["X_umap", "umap", "X_umap_orig"])
    _, right_umap = _find_umap_in_obsm(adata, ["X_umap_custom", "X_umap_new", "umap_custom"])
    
    if left_umap is None or right_umap is None:
        raise ValueError(
            "Required UMAP embeddings not found!\n"
            "Please ensure adata.obsm contains one of the following keys:\n"
            "- Original UMAP: X_umap / umap / X_umap_orig\n"
            "- Custom UMAP: X_umap_custom / X_umap_new / umap_custom"
        )
    
    # --------------------
    # 数据预处理
    # --------------------
    # 统一cell_id为字符串格式
    obs_index = adata.obs_names.astype(str)
    new_clust_df["cell_id"] = new_clust_df["cell_id"].astype(str)
    
    # 构建cell_id到cluster的映射
    cluster_map = dict(zip(new_clust_df["cell_id"], new_clust_df["cluster"]))
    
    # 构建绘图数据框
    df = pd.DataFrame(
        {
            "cell_id": obs_index,
            "lx": left_umap[:, 0],
            "ly": left_umap[:, 1],
            "rx": right_umap[:, 0],
            "ry": right_umap[:, 1],
        },
        index=obs_index,
    )
    
    # 映射簇标签并过滤无效值
    df["cluster"] = df["cell_id"].map(cluster_map)
    df = df.dropna(subset=["cluster"]).copy()
    
    # 簇标签排序（数字优先）
    def _sort_key(x):
        try:
            return int(x)
        except ValueError:
            return x
    cluster_cats = sorted(df["cluster"].unique(), key=_sort_key)
    palette = _make_palette(cluster_cats)
    
    # --------------------
    # 计算UMAP帧参数（自适应偏移，无重叠）
    # --------------------
    pad_ratio = 0.03
    
    # 计算每个UMAP的边界
    lx_min, lx_max = df["lx"].min(), df["lx"].max()
    ly_min, ly_max = df["ly"].min(), df["ly"].max()
    rx_min, rx_max = df["rx"].min(), df["rx"].max()
    ry_min, ry_max = df["ry"].min(), df["ry"].max()
    
    # 全局Y轴范围（固定帧高度）
    y_min_global = min(ly_min, ry_min)
    y_max_global = max(ly_max, ry_max)
    frame_height = y_max_global - y_min_global
    
    # 帧宽度（取两个UMAP的最大宽度）
    lx_width = lx_max - lx_min
    rx_width = rx_max - rx_min
    frame_width = max(lx_width, rx_width)
    
    # 计算内边距
    wpad = frame_width * pad_ratio
    hpad = frame_height * pad_ratio
    
    # 计算帧间距（自适应，最小0.1绝对单位）
    min_gap = 0.1
    gap = max(min_gap, frame_width * 0.25)
    
    # 计算偏移量（保证两帧不重叠）
    shift = (frame_width + 2 * wpad + gap) / 2
    df["lx_shift"] = df["lx"] - shift
    df["rx_shift"] = df["rx"] + shift
    
    # 帧中心坐标
    lx_center = (lx_min + lx_max) / 2 - shift
    rx_center = (rx_min + rx_max) / 2 + shift
    y_center = (y_min_global + y_max_global) / 2
    
    # --------------------
    # 处理高亮簇
    # --------------------
    selection = [str(s) for s in selection]  # 统一转为字符串
    sel_df = df[df["cluster"].isin(selection)].copy()
    
    # --------------------
    # 绘图
    # --------------------
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    
    # 绘制左右UMAP散点
    ax.scatter(
        df["lx_shift"], df["ly"],
        s=point_size,
        c=[palette[c] for c in df["cluster"]],
        alpha=alpha,
        linewidths=0,
        rasterized=True,
        zorder=1,
    )
    ax.scatter(
        df["rx_shift"], df["ry"],
        s=point_size,
        c=[palette[c] for c in df["cluster"]],
        alpha=alpha,
        linewidths=0,
        rasterized=True,
        zorder=1,
    )
    
    # 绘制高亮簇的跨UMAP连线
    for _, row in sel_df.iterrows():
        ax.plot(
            [row["rx_shift"], row["lx_shift"]],
            [row["ry"], row["ly"]],
            color=palette[row["cluster"]],
            alpha=line_alpha,
            linewidth=0.2,
            zorder=2,
        )
    
    # 绘制两个UMAP的边框
    for cx in (lx_center, rx_center):
        ax.add_patch(
            Rectangle(
                (
                    cx - frame_width / 2 - wpad,
                    y_center - frame_height / 2 - hpad,
                ),
                frame_width + 2 * wpad,
                frame_height + 2 * hpad,
                linewidth=1.2,
                edgecolor="black",
                facecolor="none",
                zorder=3,
            )
        )
    
    # 添加标题
    top_y = y_center + frame_height / 2 + hpad + 0.02 * frame_height
    ax.text(lx_center, top_y, "UMAP", ha="center", va="bottom", fontsize=18)
    ax.text(rx_center, top_y, "rpcUMAP", ha="center", va="bottom", fontsize=18)
    
    # 隐藏坐标轴
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    # 保存图片
    plt.tight_layout(pad=0.3)
    fig.savefig(out_file, dpi=600, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    
    print(f"[SUCCESS] UMAP comparison plot saved to: {out_file}")