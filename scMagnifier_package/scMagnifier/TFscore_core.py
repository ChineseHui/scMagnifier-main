#!/usr/bin/env python3
"""
TFscore.py

TF得分分析核心函数（package version）
精简版：仅生成每个簇的top10基因条形图，无任何CSV文件输出

Package usage:
    from scMagnifier import TFscore
    
    # 基础使用（默认读取perturb/consensus/merge的输出文件）
    TFscore()
    
    # 自定义参数使用
    TFscore(
        csv_dirs=["/path/to/perturb/0p2"],
        h5ad_path="/path/to/adata_with_rpcumap.h5ad",
        raw_matrix_csv="/path/to/original_matrix.csv",
        consensus_csv="/path/to/merged_clusters.csv"
    )
"""
import warnings
warnings.filterwarnings("ignore")

import os
import re
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
from typing import List, Dict

plt.rcParams["figure.dpi"] = 600  # 全局dpi设置为600
plt.rcParams['axes.unicode_minus'] = False  

# -----------------------------
# 必需工具函数
# -----------------------------
def find_files_starting_with(folder: str, prefix: str) -> List[str]:
    if not os.path.isdir(folder):
        return []
    return sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.startswith(prefix) and f.lower().endswith(".csv")])

def sanitize_gene_from_cluster_fname(fname: str) -> str:
    base = os.path.splitext(os.path.basename(fname))[0]
    m = re.match(r"cluster_(.+?)_(?:-?\d+\.?\d*|p?\d+)$", base)
    if m:
        return m.group(1)
    parts = base.split('_')
    if len(parts) >= 2:
        return parts[1]
    return base.replace("cluster_", "")

def sanitize_gene_from_perturbed_matrix_fname(fname: str) -> str:
    base = os.path.splitext(os.path.basename(fname))[0]
    m = re.match(r"perturbed_matrix_(.+?)_(?:-?\d+\.?\d*|p?\d+)$", base)
    if m:
        return m.group(1)
    parts = base.split('_')
    if len(parts) >= 3:
        return parts[2]
    return base.replace("perturbed_matrix_", "")

def build_overlap_matrix(ser_orig: pd.Series, ser_pert: pd.Series) -> pd.DataFrame:
    common = ser_orig.index.intersection(ser_pert.index)
    if len(common) == 0:
        raise ValueError("No overlapping cells between original and perturbed series.")
    o = ser_orig.loc[common].astype(str)
    p = ser_pert.loc[common].astype(str)
    rows = sorted(o.unique(), key=lambda x: (int(x) if x.isdigit() else x))
    cols = sorted(p.unique(), key=lambda x: (int(x) if x.isdigit() else x))
    M = pd.DataFrame(0, index=rows, columns=cols, dtype=int)
    for cell in common:
        M.at[str(o.loc[cell]), str(p.loc[cell])] += 1
    return M

def match_by_max_overlap(M: pd.DataFrame) -> Dict[str, str]:
    mapping = {}
    for r in M.index:
        if M.loc[r].sum() == 0:
            mapping[r] = None
        else:
            mapping[r] = M.loc[r].idxmax()
    return mapping

# -----------------------------
# 必需核心分析函数
# -----------------------------
def compute_binary_change_vector(adata, ser_pert: pd.Series, orig_cluster_key: str = "leiden", cells_ref: List[str] = None):
    if cells_ref is None:
        cell_order = adata.obs_names.astype(str).tolist()
    else:
        cell_order = list(cells_ref)
    n = len(cell_order)
    vec = np.zeros(n, dtype=int)
    orig_ser = adata.obs[orig_cluster_key].astype(str)
    common = orig_ser.index.intersection(ser_pert.index)
    if len(common) == 0:
        print("[WARN] No overlapping cells between adata and perturbed series; returning zeros.")
        return vec, cell_order
    M = build_overlap_matrix(orig_ser, ser_pert)
    mapping = match_by_max_overlap(M)
    idx_map = {c: i for i, c in enumerate(cell_order)}
    for cell in common:
        orig_label = str(orig_ser.loc[cell])
        pert_label = str(ser_pert.loc[cell])
        mapped = mapping.get(orig_label, None)
        if mapped is None:
            changed = 1 if pert_label != "" else 0
        else:
            changed = 1 if pert_label != mapped else 0
        vec[idx_map[cell]] = int(changed)
    return vec, cell_order

def compute_continuous_change_vector(raw_df: pd.DataFrame, pert_df: pd.DataFrame, cells_ref: List[str] = None, eps: float = 1e-8):
    if cells_ref is None:
        cell_order = list(raw_df.index.astype(str))
    else:
        cell_order = list(cells_ref)
    raw = raw_df.reindex(cell_order).copy()
    pert = pert_df.reindex(cell_order).copy()
    raw_log = np.log1p(raw.astype(float))
    pert_log = np.log1p(pert.astype(float))
    gene_mean = raw_log.mean(axis=0)
    gene_std = raw_log.std(axis=0).replace(0, eps)
    raw_stdzd = (raw_log - gene_mean) / gene_std
    pert_stdzd = (pert_log - gene_mean) / gene_std
    diff = pert_stdzd.values - raw_stdzd.values
    norms = np.linalg.norm(diff, ord=2, axis=1)
    norms = np.nan_to_num(norms, nan=0.0)
    return norms, cell_order

# -----------------------------
# 仅保留top10基因绘图函数
# -----------------------------
def plot_top_genes_barplot(gene_df, outpath, title="", topk=10):
    top_df = gene_df.head(topk).copy()

    # 直接为所有10个基因着色
    colors = ['#3686C0' for _ in range(topk)]
    top_df['color'] = colors

    plt.figure(figsize=(max(6, 0.4 * topk), 4))
    plt.bar(range(topk), top_df["gene_cluster_score"], color=top_df['color'], align="center")

    # 坐标轴刻度向外
    plt.gca().tick_params(axis='x', direction='out', length=6)
    plt.gca().tick_params(axis='y', direction='out', length=6)

    plt.xticks(range(topk), top_df.index, rotation=90, ha="center", fontsize=11)
    plt.ylabel("Gene importance")
    plt.title(title.replace("(0p1)", "").strip())
    plt.tight_layout()

    # 移除文件名中的_with_p后缀
    outpath = outpath.replace("_with_p", "")
    plt.savefig(outpath, bbox_inches="tight", dpi=600)
    plt.close()

# -----------------------------
# 精简后的核心分析流程
# -----------------------------
def analyze_one_folder(folder, h5ad_path, raw_matrix_csv, consensus_csv, outdir_base,
                       leiden_key="leiden", weight_binary=1.0, weight_continuous=1.0):

    tag = os.path.basename(folder.rstrip("/"))
    print(f"\n[INFO] ===== Processing folder: {tag} =====")
    outdir = os.path.join(outdir_base, f"analysis_{tag}")
    os.makedirs(outdir, exist_ok=True)
    analysis_dir = os.path.join(outdir, "analysis_result")
    os.makedirs(analysis_dir, exist_ok=True)

    # 加载必需数据
    adata = sc.read_h5ad(h5ad_path)
    cell_order = adata.obs_names.astype(str).tolist()
    print(f"[INFO] Loaded adata with {adata.n_obs} cells")

    raw_df = pd.read_csv(raw_matrix_csv, index_col=0)
    raw_df.index = raw_df.index.astype(str)
    raw_df = raw_df.reindex(cell_order)

    # 读取扰动相关文件
    cluster_files = find_files_starting_with(folder, "cluster_")
    pert_matrix_files = find_files_starting_with(folder, "perturbed_matrix_")
    print(f"[INFO] Found {len(cluster_files)} cluster_ files and {len(pert_matrix_files)} perturbed_matrix_ files")

    # 计算二进制扰动矩阵
    binary_rows, binary_genes = [], []
    for f in cluster_files:
        gene = sanitize_gene_from_cluster_fname(f)
        ser = pd.read_csv(f, index_col=0).iloc[:, -1]
        ser.index = ser.index.astype(str)
        vec, _ = compute_binary_change_vector(adata, ser, orig_cluster_key=leiden_key, cells_ref=cell_order)
        binary_rows.append(vec)
        binary_genes.append(gene)
    binary_mat = pd.DataFrame(np.vstack(binary_rows), index=binary_genes, columns=cell_order)

    # 计算连续扰动矩阵
    cont_rows, cont_genes = [], []
    for f in pert_matrix_files:
        gene = sanitize_gene_from_perturbed_matrix_fname(f)
        pert_df = pd.read_csv(f, index_col=0)
        pert_df.index = pert_df.index.astype(str)
        pert_df = pert_df.reindex(cell_order)
        pert_df = pert_df.reindex(columns=raw_df.columns, fill_value=0)
        norms, _ = compute_continuous_change_vector(raw_df=raw_df, pert_df=pert_df, cells_ref=cell_order)
        cont_rows.append(norms)
        cont_genes.append(gene)
    cont_mat = pd.DataFrame(np.vstack(cont_rows), index=cont_genes, columns=cell_order)

    # 保留共同基因并合并矩阵
    common_genes = [g for g in binary_genes if g in cont_mat.index]
    binary_mat = binary_mat.loc[common_genes]
    cont_mat = cont_mat.loc[common_genes]

    # 连续矩阵归一化
    cont_scaled = cont_mat.copy().astype(float)
    for g in cont_scaled.index:
        row = cont_scaled.loc[g].values
        mn, mx = np.nanmin(row), np.nanmax(row)
        cont_scaled.loc[g] = 0.0 if np.isclose(mx, mn) else (row - mn) / (mx - mn)

    # 计算综合扰动矩阵
    combined = weight_binary * binary_mat.astype(float) + weight_continuous * cont_scaled

    # 加载共识聚类数据
    cons_df = pd.read_csv(consensus_csv, index_col=0)
    cons_ser = cons_df.iloc[:, -1].astype(str)
    cons_ser.index = cons_ser.index.astype(str)
    cons_ser = cons_ser.reindex(cell_order).fillna("NA")

    # 按聚类计算基因重要性并生成top10图
    cluster_list = [cl for cl in cons_ser.unique() if cl != "NA"]
    for cl in cluster_list:
        # 筛选该聚类的细胞
        cells_in_cl = [c for c in cons_ser[cons_ser == cl].index.tolist() if c in combined.columns]
        if len(cells_in_cl) == 0:
            print(f"[WARN] Cluster {cl} has no valid cells, skipping")
            continue

        # 计算基因得分和p值
        gene_scores = combined[cells_in_cl].mean(axis=1)
        all_scores = gene_scores.values
        p_values = [(1 + np.sum(all_scores >= s)) / (1 + len(all_scores)) for s in all_scores]
        gene_pvals = pd.Series(p_values, index=gene_scores.index)

        gene_df = pd.DataFrame({
            "gene_cluster_score": gene_scores,
            "empirical_p": gene_pvals
        }).sort_values("gene_cluster_score", ascending=False)

        # 绘制top10基因图
        plot_top_genes_barplot(
            gene_df,
            os.path.join(analysis_dir, f"top10_genes_cluster_{cl}_with_p.png"),
            title=f"Top 10 genes for cluster {cl} ({tag})",
            topk=10
        )

    print(f"[DONE] Folder {tag} processed successfully -> {analysis_dir}")

# -----------------------------
# 包对外调用函数：TFscore
# -----------------------------
def TFscore(
    csv_dirs: List[str] = ["perturb_results/0p1"],  
    h5ad_path: str = os.path.join("consensus_result", "adata_with_rpcumap.h5ad"),  
    raw_matrix_csv: str = os.path.join("perturb_results", "0p1", "original_matrix.csv"),  
    consensus_csv: str = os.path.join("merged_result", "merged_clusters.csv"),  
    outdir_base: str = "TFscore_output",
    leiden_key: str = "leiden",
    weight_binary: float = 1.0,
    weight_continuous: float = 1.0
) -> None:
    """
    TF得分分析核心函数（适配perturb/consensus/merge输出）
    功能：仅生成每个簇的top10基因条形图，无任何CSV文件输出
    
    参数说明：
    ----------
    csv_dirs : List[str], optional
        包含perturb输出的cluster_/perturbed_matrix_文件的目录列表（默认：["perturb_results/0p1"]，perturb函数默认输出路径）
    h5ad_path : str, optional
        consensus输出的h5ad文件路径（默认：consensus_result/adata_with_rpcumap.h5ad，consensus函数输出）
    raw_matrix_csv : str, optional
        perturb输出的原始表达矩阵CSV路径（默认：perturb_results/0p1/original_matrix.csv，perturb函数输出）
    consensus_csv : str, optional
        merge输出的合并聚类CSV路径（默认：merged_result/merged_clusters.csv，merge函数输出）
    outdir_base : str, optional
        TFscore结果输出根目录（默认：TFscore_output）
    leiden_key : str, optional
        原始聚类列名（默认：leiden）
    weight_binary : float, optional
        二进制扰动得分权重（默认：1.0）
    weight_continuous : float, optional
        连续扰动得分权重（默认：1.0）
    
    返回值：
    ----------
    None
        结果保存到指定输出目录，仅生成各聚类的top10基因条形图。
    """
    # 校验关键文件/目录存在性
    for dir_path in csv_dirs:
        if not os.path.isdir(dir_path):
            raise NotADirectoryError(f"[ERROR] Directory not found: {dir_path}\nHint: Check if perturb function has run successfully.")
    
    for file_path, hint in [
        (h5ad_path, "consensus"),
        (raw_matrix_csv, "perturb"),
        (consensus_csv, "merge")
    ]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"[ERROR] File not found: {file_path}\n"
                f"Hint: Check if {hint} function has run successfully, or confirm the file path is correct."
            )

    # 遍历处理每个目录
    for folder in csv_dirs:
        analyze_one_folder(
            folder=folder,
            h5ad_path=h5ad_path,
            raw_matrix_csv=raw_matrix_csv,
            consensus_csv=consensus_csv,
            outdir_base=outdir_base,
            leiden_key=leiden_key,
            weight_binary=weight_binary,
            weight_continuous=weight_continuous
        )
    
    print(f"\n[INFO] All TFscore analysis completed -> {outdir_base}")