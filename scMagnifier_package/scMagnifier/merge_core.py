#!/usr/bin/env python3
"""
merge.py

簇合并与可视化核心函数（package version）
用途:
 - 从高分辨率聚类 CSV（cell_id,cluster）和 h5ad 出发，
   使用高变基因计算簇质心，根据簇间质心距离中位数（THM）合并簇（一次），
   再把仍小于阈值（默认 1% 总细胞数，至少 1 个细胞）的簇合并到最近簇（迭代），
   最后输出合并后的 CSV 与在 old/new UMAP 的可视化图。

说明（与 preprocess_for_celloracle.py 配合）:
 - preprocess_for_celloracle.py 已经将 AnnData subset 到 HVG，
   并做了 normalize_total -> log1p -> scale。此脚本直接使用 adata_sub.X 作为质心表达矩阵，
   **不再**重新计算 HVG 或重复 normalize/log1p。

Package usage:
    from scMagnifier import merge
    
    # 基础使用（自动查找consensus_result下louvain/leiden_res开头的CSV）
    merge()
    
    # 自定义参数使用
    merge(
        h5ad_path="/path/to/adata_with_custom_umap.h5ad",
        cluster_csv="/path/to/louvain_res0.3.csv",  # 支持louvain/leiden任意分辨率
        outdir="my_merged_result",
        min_size_fraction=0.01
    )
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
from typing import List, Tuple, Optional  



# -----------------------------
# Helper I/O & utils
# -----------------------------
def read_cluster_csv_as_series(path: str) -> pd.Series:
    """读取单个 cluster CSV 并返回 Series (index=cell_id, values=cluster as str)"""
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
    new_vals = []
    for v in series.values.astype(str):
        if v in mapping:
            new_vals.append(mapping[v] + start_label_offset)
        else:
            new_vals.append(-1)
    return pd.Series(new_vals, index=series.index)

def merge_small_clusters_iteratively(labels_array: np.ndarray, coords: np.ndarray, min_size: int) -> np.ndarray:
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

def find_cluster_csv(consensus_dir: str = "consensus_result") -> str:
    """
    自动查找consensus_result目录下的聚类CSV文件
    匹配规则：louvain_res*.csv 或 leiden_res*.csv（任意分辨率）
    """
    # 定义两种聚类方法的匹配模式
    patterns = [
        os.path.join(consensus_dir, "louvain_res*.csv"),
        os.path.join(consensus_dir, "leiden_res*.csv")
    ]
    
    # 收集所有匹配的文件
    matching_files = []
    for pattern in patterns:
        matching_files.extend(glob.glob(pattern, recursive=False))
    
    if not matching_files:
        raise FileNotFoundError(
            f"[ERROR] No cluster CSV files found in {consensus_dir}!\n"
            f"Expected files: louvain_res*.csv or leiden_res*.csv (e.g., louvain_res0.3.csv, leiden_res1.0.csv)\n"
            f"Hint: Check if consensus function has run, or specify cluster_csv manually."
        )
    
    # 按文件名排序，取第一个匹配文件（保证结果可复现）
    matching_files.sort()
    selected_file = matching_files[0]
    print(f"[INFO] Auto-selected cluster CSV: {selected_file}")
    return selected_file

# -----------------------------
# Main flow
# -----------------------------
def merge(
    h5ad_path: str = os.path.join("consensus_result", "adata_with_rpcumap.h5ad"),  
    cluster_csv: Optional[str] = None,  # 默认None，自动查找louvain/leiden_res开头的CSV
    outdir: str = "merged_result",
    min_size_fraction: float = 0.01,
    n_top_hvg: int = 2000,
    th_scaler: float = 0.75
) -> None:
    """
    簇合并与可视化核心函数（适配consensus输出）
    
    参数说明：
    ----------
    h5ad_path : str, optional
        包含自定义UMAP的h5ad文件路径（默认：consensus_result/adata_with_custom_umap.h5ad，consensus函数默认输出）
    cluster_csv : str | None, optional
        高分辨率聚类结果CSV路径：
        - None（默认）：自动查找consensus_result目录下louvain_res*.csv/leiden_res*.csv
        - 字符串：使用指定的CSV文件路径（支持louvain/leiden任意分辨率）
    outdir : str, optional
        簇合并结果输出目录（默认：merged_result）
    min_size_fraction : float, optional
        最小簇大小占总细胞数的比例（默认：0.01，即1%）
    n_top_hvg : int, optional
        高变基因数量（默认：2000，与preprocess保持一致）
    th_scaler : float, optional
        THM（质心距离中位数）缩放系数（默认：0.75）
    
    注意：
    ------
    1. 若你在consensus中修改了输出文件名（如adata_with_rpcumap.h5ad），请同步修改h5ad_path默认值；
    2. 此函数直接使用adata.X（HVG+标准化后）计算质心，不重复预处理；
    3. 自动查找逻辑优先匹配louvain_res*.csv，若无则匹配leiden_res*.csv（按文件名排序）。
    
    返回值：
    ----------
    None
        结果保存到指定输出目录，包含合并后的聚类CSV、old/new UMAP可视化图。
    """
    os.makedirs(outdir, exist_ok=True)

    # -----------------------------
    # 处理cluster_csv：自动查找或使用指定路径
    # -----------------------------
    if cluster_csv is None:
        # 自动查找consensus_result下的louvain/leiden_res开头的CSV
        cluster_csv = find_cluster_csv(consensus_dir=os.path.dirname(h5ad_path))

    # -----------------------------
    # 校验关键文件是否存在
    # -----------------------------
    for file_path, hint in [(h5ad_path, "consensus"), (cluster_csv, "cluster CSV")]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"[ERROR] File not found: {file_path}\n"
                f"Hint: Check if {hint} function has run successfully, or confirm the file path is correct."
            )

    print("[INFO] Loading h5ad and cluster CSV ...")
    adata = sc.read_h5ad(h5ad_path)
    csv_ser = read_cluster_csv_as_series(cluster_csv)
    print(f"[INFO] Read cluster CSV with {csv_ser.shape[0]} rows.")

    canonical = adata.obs_names.astype(str).tolist()
    intersect = [c for c in canonical if c in set(csv_ser.index)]
    if len(intersect) == 0:
        raise ValueError("No overlap between h5ad cells and cluster CSV cell IDs.")
    print(f"[INFO] {len(intersect)} cells after alignment (intersection).")

    adata_sub = adata[intersect, :].copy()
    csv_ser_sub = csv_ser.reindex(intersect).astype(str)

    unique_input_labels = pd.Series(csv_ser_sub.values).astype('category').cat.categories.tolist()
    old_to_code = {str(lbl): idx for idx, lbl in enumerate(unique_input_labels)}
    code_to_old = {v: k for k, v in old_to_code.items()}
    labels_initial = np.array([old_to_code[s] for s in csv_ser_sub.values.astype(str)], dtype=int)
    print(f"[INFO] Found {len(unique_input_labels)} initial D-clusters.")

    if adata_sub.X is None:
        raise ValueError("adata_sub has no expression matrix (adata_sub.X is None).")
    if hasattr(adata_sub.X, "toarray"):
        coords = np.asarray(adata_sub.X.toarray())
    else:
        coords = np.asarray(adata_sub.X)
    print(f"[INFO] Centroid coords shape (cells x features): {coords.shape}")

    centroids, cluster_ids = compute_centroids(coords, labels_initial)
    print(f"[INFO] Computed {centroids.shape[0]} centroids based on HVG expression.")

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

    merged_series = relabel_series_by_mapping(csv_ser_sub, mapping_after_merge, start_label_offset=0)
    uniq_after = np.unique(merged_series.values)
    remap = {old: new for new, old in enumerate(uniq_after)}
    merged_labels = np.array([remap[v] for v in merged_series.values], dtype=int)
    print(f"[INFO] {len(np.unique(merged_labels))} clusters after the first merge step.")

    n_cells = merged_labels.shape[0]
    min_size = max(1, int(np.ceil(min_size_fraction * n_cells)))
    print(f"[INFO] Minimum cluster size threshold = {min_size} cells ({min_size_fraction*100:.2f}%).")
    merged_labels_final = merge_small_clusters_iteratively(merged_labels, coords, min_size)
    print(f"[INFO] {len(np.unique(merged_labels_final))} clusters after merging small clusters.")

    out_csv = os.path.join(outdir, "merged_clusters.csv")
    df_out = pd.DataFrame({"cell_id": intersect, "cluster": merged_labels_final})
    df_out.to_csv(out_csv, index=False)
    print(f"[SAVED] Merged cluster CSV -> {out_csv}")

    # -----------------------------
    # Plotting helper 
    # -----------------------------
    def plot_umap(adata_tmp, color_key, png_path):
        fig = sc.pl.umap(adata_tmp, color=color_key, size=8, alpha=0.8, show=False, return_fig=True)
        fig.axes[0].set_title(fig.axes[0].get_title(), fontsize=10)
        legend = fig.axes[0].get_legend()
        if legend:
            for text in legend.get_texts():
                text.set_fontsize(10)
        fig.savefig(png_path, bbox_inches='tight')
        plt.close(fig)

    # old UMAP
    if 'X_umap' in adata.obsm:
        adata_plot_old = adata[intersect, :].copy()
        adata_plot_old.obs['merged_cluster'] = merged_labels_final.astype(str)
        plot_umap(adata_plot_old, 'merged_cluster', os.path.join(outdir, "umap_merged.png"))
        print(f"[SAVED] Old UMAP merged plot -> {os.path.join(outdir, 'umap_merged.png')}")
    else:
        print("[INFO] Original adata has no 'X_umap'; skipping old UMAP plot.")

    # new/custom UMAP
    if 'X_umap_custom' in adata_sub.obsm:
        adata_plot_new = adata_sub.copy()
        adata_plot_new.obs['merged_cluster'] = merged_labels_final.astype(str)
        adata_plot_new.obsm['X_umap'] = adata_plot_new.obsm['X_umap_custom']
        plot_umap(adata_plot_new, 'merged_cluster', os.path.join(outdir, "rpcumap_merged.png"))
        print(f"[SAVED] New UMAP merged plot -> {os.path.join(outdir, 'rpcumap_merged.png')}")
    else:
        print("[INFO] adata_sub has no 'X_umap_custom'; skipping new UMAP plot.")

    print("[DONE] Merge + visualization complete.")
