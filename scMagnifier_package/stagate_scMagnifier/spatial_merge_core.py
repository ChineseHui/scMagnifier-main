#!/usr/bin/env python3
"""
spatial_merge.py

功能：
- 聚类结果合并（基于质心距离+小聚类合并）
- 支持UMAP可视化（旧/新）
- 支持空间图绘制（可配置spot_size，与STAGATE Visium风格一致）
- 默认参数自动匹配consensus_result目录下leiden_res开头的CSV文件
"""

import warnings
warnings.filterwarnings("ignore")

import os
import glob
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
from scipy.sparse.csgraph import connected_components
from typing import List, Tuple, Optional  # 新增Optional

# =============================
# Helper I/O & utils
# =============================
def read_cluster_csv_as_series(path: str) -> pd.Series:
    """读取聚类CSV文件并返回Series（索引为细胞ID，值为聚类标签）"""
    try:
        df = pd.read_csv(path, index_col=0)
    except Exception:
        df = pd.read_csv(path, header=None)
        df = df.set_index(0)
    if df.shape[1] > 1:
        ser = df.iloc[:, -1].astype(str)
    else:
        ser = df.iloc[:, 0].astype(str)
    ser.index = ser.index.astype(str)
    return ser

def compute_centroids(coords: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """计算每个聚类的质心坐标"""
    unique = np.unique(labels)
    centroids = []
    for u in unique:
        mask = labels == u
        if mask.sum() == 0:
            centroids.append(np.zeros((coords.shape[1],)))
        else:
            centroids.append(coords[mask].mean(axis=0))
    centroids = np.vstack(centroids) if len(centroids) > 0 else np.zeros((0, coords.shape[1]))
    return centroids, unique

def merge_by_threshold(centroids: np.ndarray, cluster_ids: np.ndarray, thm: float) -> dict:
    """基于距离阈值合并聚类"""
    mapping = {}
    if centroids.shape[0] == 0:
        return mapping
    D = pairwise_distances(centroids, metric='euclidean')
    np.fill_diagonal(D, np.inf)
    adj = (D < thm).astype(int)
    n_comp, labels_comp = connected_components(csgraph=adj, directed=False, connection='weak')
    for idx, old in enumerate(cluster_ids):
        mapping[str(old)] = int(labels_comp[idx])
    return mapping

def relabel_series_by_mapping(series: pd.Series, mapping: dict, start_label_offset: int = 0) -> pd.Series:
    """根据映射关系重新标记聚类标签"""
    new_vals = []
    for v in series.values.astype(str):
        if v in mapping:
            new_vals.append(mapping[v] + start_label_offset)
        else:
            new_vals.append(-1)
    return pd.Series(new_vals, index=series.index)

def merge_small_clusters_iteratively(labels_array: np.ndarray, coords: np.ndarray, min_size: int) -> np.ndarray:
    """迭代合并小聚类（小于最小尺寸阈值）"""
    labels = labels_array.copy().astype(int)
    while True:
        uniq, counts = np.unique(labels, return_counts=True)
        small_mask = counts < min_size
        if not small_mask.any():
            break
        small_clusters = uniq[small_mask]
        centroids, cluster_ids = compute_centroids(coords, labels)
        id_to_idx = {cid: idx for idx, cid in enumerate(cluster_ids)}
        for scid in small_clusters:
            sc_idx = id_to_idx[scid]
            dists = np.linalg.norm(centroids - centroids[sc_idx:sc_idx+1], axis=1).reshape(-1)
            dists[sc_idx] = np.inf
            nearest_idx = int(np.argmin(dists))
            nearest_cid = cluster_ids[nearest_idx]
            labels[labels == scid] = nearest_cid
    uniq_final = np.unique(labels)
    mapping = {old: new for new, old in enumerate(uniq_final)}
    labels_mapped = np.array([mapping[x] for x in labels], dtype=int)
    return labels_mapped

def find_leiden_res_csv(consensus_dir: str = "consensus_result") -> str:
    """自动查找consensus_result目录下以leiden_res开头的CSV文件"""
    # 匹配模式：leiden_res任意字符.csv（不区分大小写）
    pattern = os.path.join(consensus_dir, "leiden_res*.csv")
    matching_files = glob.glob(pattern, recursive=False)
    
    if not matching_files:
        raise FileNotFoundError(
            f"No leiden_res*.csv files found in {consensus_dir}!\n"
            f"Hint: Check if consensus clustering has run, or specify cluster_csv manually."
        )
    
    # 取第一个匹配的文件（按名称排序）
    matching_files.sort()
    selected_file = matching_files[0]
    print(f"[INFO] Auto-selected cluster CSV: {selected_file}")
    return selected_file

# =============================
# Main flow: spatial_merge 核心函数
# =============================
def spatial_merge(
    h5ad_path: str = os.path.join("consensus_result", "adata_with_rpcumap.h5ad"),
    cluster_csv: Optional[str] = None,  # 默认None，自动查找
    outdir: str = "merged_result",
    min_size_fraction: float = 0.01,
    n_top_hvg: int = 2000,
    th_scaler: float = 0.75,
    spot_size: Optional[int] = None  # 新增：空间图spot_size参数，默认None（使用scanpy默认值）
) -> None:
    """
    空间转录组聚类结果合并与可视化函数
    
    Parameters
    ----------
    h5ad_path : str, optional
        输入h5ad文件路径（默认：consensus_result/adata_with_rpcumap.h5ad）
    cluster_csv : str | None, optional
        输入聚类结果CSV文件路径：
        - None（默认）：自动查找consensus_result目录下leiden_res开头的CSV文件
        - 字符串：使用指定的CSV文件路径
    outdir : str, optional
        结果输出目录（默认：merged_result）
    min_size_fraction : float, optional
        小聚类合并的最小尺寸比例（占总细胞数的比例），默认0.01
    n_top_hvg : int, optional
        高可变基因数量（用于质心计算），默认2000
    th_scaler : float, optional
        距离阈值缩放系数，默认1.0
    spot_size : Optional[int], optional
        空间图的spot大小：
        - 传入数值：使用该值绘制空间图
        - 不传/None：使用scanpy默认值
    """
    os.makedirs(outdir, exist_ok=True)

    # -----------------------------
    # 自动处理cluster_csv（核心修改）
    # -----------------------------
    if cluster_csv is None:
        # 自动查找leiden_res开头的CSV文件
        cluster_csv = find_leiden_res_csv(consensus_dir=os.path.dirname(h5ad_path))
    
    # -----------------------------
    # 加载并校验数据
    # -----------------------------
    print("[INFO] Loading h5ad and cluster CSV ...")
    # 校验输入文件是否存在
    if not os.path.exists(h5ad_path):
        raise FileNotFoundError(f"h5ad file not found: {h5ad_path}")
    if not os.path.exists(cluster_csv):
        raise FileNotFoundError(f"Cluster CSV file not found: {cluster_csv}")
    
    adata = sc.read_h5ad(h5ad_path)
    csv_ser = read_cluster_csv_as_series(cluster_csv)
    print(f"[INFO] Read cluster CSV with {csv_ser.shape[0]} rows.")

    # 细胞ID对齐
    canonical = adata.obs_names.astype(str).tolist()
    intersect = [c for c in canonical if c in set(csv_ser.index)]
    if len(intersect) == 0:
        raise ValueError("No overlap between h5ad cells and cluster CSV cell IDs.")
    print(f"[INFO] {len(intersect)} cells after alignment (intersection).")

    adata_sub = adata[intersect, :].copy()
    csv_ser_sub = csv_ser.reindex(intersect).astype(str)

    # 初始聚类标签编码
    unique_input_labels = pd.Series(csv_ser_sub.values).astype('category').cat.categories.tolist()
    old_to_code = {str(lbl): idx for idx, lbl in enumerate(unique_input_labels)}
    code_to_old = {v: k for k, v in old_to_code.items()}
    labels_initial = np.array([old_to_code[s] for s in csv_ser_sub.values.astype(str)], dtype=int)
    print(f"[INFO] Found {len(unique_input_labels)} initial D-clusters.")

    # 提取表达矩阵用于质心计算
    if adata_sub.X is None:
        raise ValueError("adata_sub has no expression matrix (adata_sub.X is None).")
    if hasattr(adata_sub.X, "toarray"):
        coords = np.asarray(adata_sub.X.toarray())
    else:
        coords = np.asarray(adata_sub.X)
    print(f"[INFO] Centroid coords shape (cells x features): {coords.shape}")

    # 计算初始聚类质心
    centroids, cluster_ids = compute_centroids(coords, labels_initial)
    print(f"[INFO] Computed {centroids.shape[0]} centroids based on HVG expression.")

    # 基于距离阈值合并聚类
    if centroids.shape[0] <= 1:
        print("[WARN] Only one cluster present; skipping threshold merging.")
        mapping_after_merge = {str(code_to_old[cid]): 0 for cid in cluster_ids}
    else:
        Dc = pairwise_distances(centroids, metric='euclidean')
        np.fill_diagonal(Dc, np.inf)
        min_to_nn = Dc.min(axis=1)
        THM = float(np.median(min_to_nn))
        THM_scaled = THM * th_scaler
        print(f"[INFO] THM raw = {THM:.6g}, th_scaler = {th_scaler}, THM_scaled = {THM_scaled:.6g}")

        mapping_codes = merge_by_threshold(centroids, cluster_ids, THM_scaled)
        mapping_after_merge = {}
        for old_code_str, merged_idx in mapping_codes.items():
            old_code = int(old_code_str)
            old_label_str = code_to_old[old_code]
            mapping_after_merge[str(old_label_str)] = int(merged_idx)
        print(f"[INFO] After threshold merging, {len(set(mapping_after_merge.values()))} merged clusters created.")

    # 重新标记聚类标签
    merged_series = relabel_series_by_mapping(csv_ser_sub, mapping_after_merge, start_label_offset=0)
    uniq_after = np.unique(merged_series.values)
    remap = {old: new for new, old in enumerate(uniq_after)}
    merged_labels = np.array([remap[v] for v in merged_series.values], dtype=int)
    print(f"[INFO] {len(np.unique(merged_labels))} clusters after the first merge step.")

    # 合并小聚类
    n_cells = merged_labels.shape[0]
    min_size = max(1, int(np.ceil(min_size_fraction * n_cells)))
    print(f"[INFO] Minimum cluster size threshold = {min_size} cells ({min_size_fraction*100:.2f}%).")
    merged_labels_final = merge_small_clusters_iteratively(merged_labels, coords, min_size)
    print(f"[INFO] {len(np.unique(merged_labels_final))} clusters after merging small clusters.")

    # 保存合并后的聚类结果
    out_csv = os.path.join(outdir, "merged_clusters.csv")
    df_out = pd.DataFrame({"cell_id": intersect, "cluster": merged_labels_final})
    df_out.to_csv(out_csv, index=False)
    print(f"[SAVED] Merged cluster CSV -> {out_csv}")

    # -----------------------------
    # Plotting UMAP (保持原逻辑)
    # -----------------------------
    def plot_umap(adata_tmp, color_key, png_path):
        """UMAP绘图辅助函数（统一风格）"""
        fig = sc.pl.umap(adata_tmp, color=color_key, size=8, alpha=0.8, show=False, return_fig=True)
        fig.axes[0].set_title(fig.axes[0].get_title(), fontsize=10)
        ax = fig.axes[0]
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.0)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        ax.set_xlabel("")
        ax.set_ylabel("")
        legend = fig.axes[0].get_legend()
        if legend:
            for text in legend.get_texts():
                text.set_fontsize(10)
        fig.savefig(png_path, bbox_inches='tight', dpi=600)
        plt.close(fig)

    # 旧UMAP可视化
    if 'X_umap' in adata.obsm:
        adata_plot_old = adata[intersect, :].copy()
        adata_plot_old.obs['merged_cluster'] = merged_labels_final.astype(str)
        plot_umap(adata_plot_old, 'merged_cluster', os.path.join(outdir, "umap_old_merged.png"))
        print(f"[SAVED] Old UMAP merged plot -> {os.path.join(outdir, 'umap_old_merged.png')}")

    # 新/custom UMAP可视化
    if 'X_umap_custom' in adata_sub.obsm:
        adata_plot_new = adata_sub.copy()
        adata_plot_new.obs['merged_cluster'] = merged_labels_final.astype(str)
        adata_plot_new.obsm['X_umap'] = adata_plot_new.obsm['X_umap_custom']
        plot_umap(adata_plot_new, 'merged_cluster', os.path.join(outdir, "umap_new_merged.png"))
        print(f"[SAVED] New UMAP merged plot -> {os.path.join(outdir, 'umap_new_merged.png')}")

    # -----------------------------
    # 空间作图（动态控制spot_size参数）
    # -----------------------------
    if 'spatial' in adata_sub.obsm:
        adata_sub.obs['merged_cluster'] = merged_labels_final.astype(str)
        
        # 动态构建空间作图参数
        spatial_kwargs = {
            "adata": adata_sub,
            "color": "merged_cluster",
            "show": False
        }
        # 仅当spot_size不为None时添加该参数
        if spot_size is not None:
            spatial_kwargs["spot_size"] = spot_size
            print(f"[INFO] Plotting spatial with custom spot_size={spot_size}")
        else:
            print(f"[INFO] Plotting spatial with scanpy default spot_size")
        
        # 绘制并保存空间图
        sc.pl.spatial(** spatial_kwargs)
        plt.savefig(
            os.path.join(outdir, "spatial_merged.png"),
            dpi=600,
            bbox_inches="tight"
        )
        plt.close()
        print(f"[SAVED] Spatial merged plot -> {os.path.join(outdir, 'spatial_merged.png')}")
    else:
        print("[INFO] adata_sub has no 'spatial' coordinates; skipping spatial plot.")

    print("[DONE] Merge + UMAP + spatial visualization complete.")
