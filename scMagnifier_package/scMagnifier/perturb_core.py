#!/usr/bin/env python3
"""
perturb.py

Perturbation script for CellOracle (package version)
- Supports multiple multipliers
- Uses cluster_key from input AnnData / args
- Downstream analysis: log1p -> scale -> PCA -> neighbors -> UMAP -> clustering
- UMAP plotting style unified to seaborn husl palette (size=8, alpha=0.8, dpi=600)
- label_key parameter for cell subtype (default: Cell_subtype)

Package usage:
    from scMagnifier import perturb
    
    # 基础使用（默认读取preprocess/GRN的输出文件）
    perturb()
    
    # 自定义参数使用
    perturb(
        input="/path/to/preprocessed.h5ad",
        oracle_h5="/path/to/preadata.final.celloracle.oracle",
        label_key="Cell_type",
        multiplier="0.1,-0.1"
    )
"""
import warnings
warnings.filterwarnings("ignore")


import os
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import celloracle as co
from pathlib import Path  


# -----------------------------
# Helper: extract coef_matrix per cluster 
# -----------------------------
def extract_coef_dict(oracle):
    for attr in ["coef_matrix_per_cluster", "coef_dict", "coefs", "coef_mtx_dict"]:
        if hasattr(oracle, attr):
            coef_dict = getattr(oracle, attr)
            print(f"[INFO] Found coefficient matrix in oracle.{attr}")
            return coef_dict
    raise AttributeError("Cannot find coef matrix in oracle object.")


# -----------------------------
# Helper: propagate perturbation via GRN 
# -----------------------------
def propagate_matrix(deltaX, coef_mat, pert_indices, n_iters=3):
    v = deltaX.copy()
    for _ in range(n_iters):
        v_next = v @ coef_mat
        v_next[:, pert_indices] = deltaX[:, pert_indices]  # 保持扰动基因不变
        v = v_next
    return v


# -----------------------------
# Core function for one multiplier 
# -----------------------------
def run_for_one_multiplier(args, multiplier, label_key):
    """内部核心函数（适配label_key参数）"""
    multiplier_str = str(multiplier).replace('.', 'p').replace('-', 'neg')
    outdir = os.path.join(args["outdir"], multiplier_str)
    os.makedirs(outdir, exist_ok=True)

    print(f"\n[INFO] === Running for multiplier={multiplier} ===")
    print(f"[INFO] Results will be saved under: {outdir}")

    # Step 1. Load data 
    adata = sc.read_h5ad(args["input"])
    oracle = co.load_hdf5(args["oracle_h5"])
    coef_dict = extract_coef_dict(oracle)

    # 使用 raw_count 层作为原始计数矩阵
    if "raw_count" in adata.layers:
        X_raw = adata.layers["raw_count"].copy()
    else:
        X_raw = adata.X.copy()
        print("[WARN] adata.layers['raw_count'] not found; using adata.X as raw counts.")

    gene_names = list(adata.var_names)
    cluster_labels = adata.obs[args["cluster_key"]].astype(str)
    clusters = sorted(cluster_labels.unique())
    print(f"[INFO] Found {len(clusters)} clusters in {args['cluster_key']}: {clusters}")

    # 保存原始矩阵
    raw_csv_path = os.path.join(outdir, "original_matrix.csv")
    pd.DataFrame(X_raw, columns=gene_names, index=adata.obs_names).to_csv(raw_csv_path)
    print(f"[SAVED] Original expression matrix: {raw_csv_path}")

    old_umap = adata.obsm["X_umap"].copy() if "X_umap" in adata.obsm else None

    # Step 2. Read gene list 
    with open(args["gene_list"], "r") as f:
        genes_to_perturb = [line.strip() for line in f if line.strip()]
    print(f"[INFO] Loaded {len(genes_to_perturb)} genes from txt file.")

    multiplier = float(multiplier)

    # Step 3. Loop over genes 
    for gene in genes_to_perturb:
        print("=" * 80)
        print(f"[RUNNING] Perturbing gene: {gene} (multiplier={multiplier})")

        if gene not in gene_names:
            print(f"[WARN] Gene {gene} not found in adata.var_names, skipping.")
            continue

        pert_idx = gene_names.index(gene)
        X_new_total = X_raw.copy()

        for cid in clusters:
            cells_idx = cluster_labels == str(cid)
            n_cells = np.sum(cells_idx)
            if n_cells == 0:
                continue

            if str(cid) not in coef_dict:
                print(f"[WARN] Cluster {cid} not found in oracle. Skipping.")
                continue

            coef_mat = np.array(coef_dict[str(cid)])
            X_cluster = X_raw[cells_idx, :]

            # 构建扰动矩阵
            deltaX = np.zeros_like(X_cluster)
            deltaX[:, pert_idx] = X_cluster[:, pert_idx] * multiplier

            # 扰动传播
            deltaX_prop = propagate_matrix(deltaX, coef_mat, [pert_idx], n_iters=3)
            X_new_total[cells_idx, :] += deltaX_prop

        # 保证非负
        X_new_total = np.maximum(X_new_total, 0)

        # 保存扰动结果
        perturbed_csv_path = os.path.join(outdir, f"perturbed_matrix_{gene}_{multiplier}.csv")
        pd.DataFrame(X_new_total, columns=gene_names, index=adata.obs_names).to_csv(perturbed_csv_path)
        print(f"[SAVED] Perturbed matrix for {gene}: {perturbed_csv_path}")

        # 下游分析
        adata_tmp = adata.copy()
        adata_tmp.X = X_new_total.copy()

        print("[INFO] Running log1p + scale + PCA + neighbors + UMAP + clustering ...")
        sc.pp.log1p(adata_tmp)
        sc.pp.scale(adata_tmp, max_value=10)
        sc.tl.pca(adata_tmp, svd_solver="arpack", n_comps=args["n_pcs"])
        sc.pp.neighbors(adata_tmp, n_neighbors=args["n_neighbors"], n_pcs=args["n_pcs"], random_state=args["seed"])
        sc.tl.umap(adata_tmp, random_state=args["seed"])

        # 聚类 
        if args["cluster_key"].lower() == "leiden":
            sc.tl.leiden(adata_tmp, resolution=args["resolution"],
                         key_added="leiden_perturbed", random_state=args["seed"])
            pert_cluster_key = "leiden_perturbed"

        elif args["cluster_key"].lower() == "louvain":
            sc.tl.louvain(adata_tmp, resolution=args["resolution"],
                          key_added="louvain_perturbed", random_state=args["seed"])
            pert_cluster_key = "louvain_perturbed"

        else:
            raise ValueError("cluster_key must be either 'leiden' or 'louvain'")

        adata_tmp.obsm["X_umap_perturbed"] = adata_tmp.obsm["X_umap"].copy()

        # 保存聚类结果
        cluster_csv = os.path.join(outdir, f"cluster_{gene}_{multiplier}.csv")
        adata_tmp.obs[[pert_cluster_key]].to_csv(cluster_csv)
        print(f"[SAVED] {cluster_csv}")

        if args["save_h5ad"]:
            h5ad_perturbed = os.path.join(outdir, f"adata_perturbed_{gene}_{multiplier}.h5ad")
            adata_tmp.write(h5ad_perturbed)
            print(f"[SAVED] Perturbed AnnData: {h5ad_perturbed}")
        else:
            print("[INFO] Skipped saving perturbed AnnData (save_h5ad=False)")

        if args["plot_umap"]:
            try:
                fig1 = sc.pl.umap(
                    adata_tmp,
                    color=pert_cluster_key,
                    size=8,
                    alpha=0.8,
                    show=False,
                    return_fig=True,
                    title=f"perturb on new umap"
                )
                fig1.savefig(
                    os.path.join(outdir, f"umap_new_{pert_cluster_key}_{gene}_{multiplier}.png"),
                    bbox_inches="tight"
                )
                plt.close(fig1)
            except Exception as e:
                print(f"[ERROR] Failed to plot UMAP for {pert_cluster_key}: {e}")

            if label_key in adata_tmp.obs.columns:
                try:
                    fig2 = sc.pl.umap(
                        adata_tmp,
                        color=label_key,
                        size=8,
                        alpha=0.8,
                        show=False,
                        return_fig=True,
                        title=f"{label_key} on new umap"
                    )
                    fig2.savefig(
                        os.path.join(outdir, f"umap_{label_key}_{gene}_{multiplier}.png"),
                        bbox_inches="tight"
                    )
                    plt.close(fig2)
                except Exception as e:
                    print(f"[ERROR] Failed to plot UMAP for {label_key}: {e}")
            else:
                print(f"[WARN] adata.obs['{label_key}'] not found, skipping UMAP by {label_key}.")

            if old_umap is not None:
                try:
                    adata_tmp.obsm["X_umap"] = old_umap.copy()
                    adata_tmp.obs["pert_cluster"] = adata_tmp.obs[pert_cluster_key].astype(str)

                    fig3 = sc.pl.umap(
                        adata_tmp,
                        color="pert_cluster",
                        size=8,
                        alpha=0.8,
                        show=False,
                        return_fig=True,
                        title=f"perturb on original umap"
                    )
                    fig3.savefig(
                        os.path.join(outdir, f"umap_original_on_new_{pert_cluster_key}_{gene}_{multiplier}.png"),
                        bbox_inches="tight"
                    )
                    plt.close(fig3)

                    adata_tmp.obsm["X_umap"] = adata_tmp.obsm["X_umap_perturbed"].copy()

                except Exception as e:
                    print(f"[ERROR] Failed to plot original UMAP on new {pert_cluster_key}: {e}")

        print(f"[DONE] Perturbation for {gene} completed.")
        print("=" * 80)

    print(f"\n[INFO] All genes done for multiplier={multiplier}.")


# -----------------------------
# Main package function: perturb 
# -----------------------------
def perturb(
    input: str = os.path.join("preprocessed_result", "preprocessed.h5ad"),  
    oracle_h5: str = os.path.join("GRN", "preadata.final.celloracle.oracle"), 
    cluster_key: str = "leiden",
    gene_list: str = os.path.join("GRN", "TF_HVG_intersect_genes.txt"), 
    multiplier: str = "0.1,-0.1",
    n_pcs: int = 20,
    n_neighbors: int = 10,
    resolution: float = 0.75,
    seed: int = 42,
    outdir: str = "perturb_results",
    plot_umap: bool = True,
    save_h5ad: bool = False,
    label_key: str = "Cell_subtype" 
) -> None:
    """
    参数说明：
    ----------
    input : str, optional
        预处理后的h5ad文件路径（默认：preprocessed_result/preprocessed.h5ad，preprocess函数默认输出）
    oracle_h5 : str, optional
        GRN拟合后的Oracle文件路径（默认：GRN/preadata.final.celloracle.oracle，GRN函数默认输出）
    cluster_key : str, optional
        聚类列名（默认：leiden），仅支持leiden/louvain
    gene_list : str, optional
        扰动基因列表路径（默认：GRN/TF_HVG_intersect_genes.txt，GRN函数生成的TF交集文件）
    multiplier : str, optional
        扰动倍数列表（逗号分隔，默认："0.1,-0.1"）
    n_pcs : int, optional
        PCA维度数（默认：20）
    n_neighbors : int, optional
        近邻数（默认：10）
    resolution : float, optional
        聚类分辨率（默认：0.75）
    seed : int, optional
        随机种子（默认：42）
    outdir : str, optional
        扰动结果输出目录（默认：perturb_results）
    plot_umap : bool, optional
        是否绘制UMAP图（默认：True）
    save_h5ad : bool, optional
        是否保存扰动后的h5ad文件（默认：False）
    label_key : str, optional
        细胞亚型列名（默认：Cell_subtype），无此列则跳过对应UMAP绘图
    
    返回值：
    ----------
    None
        结果保存到指定输出目录，包含扰动矩阵、聚类结果、UMAP图等
    """
    # 封装参数为字典（适配内部函数）
    args = {
        "input": input,
        "oracle_h5": oracle_h5,
        "cluster_key": cluster_key,
        "gene_list": gene_list,
        "n_pcs": n_pcs,
        "n_neighbors": n_neighbors,
        "resolution": resolution,
        "seed": seed,
        "outdir": outdir,
        "plot_umap": plot_umap,
        "save_h5ad": save_h5ad
    }

    # 校验关键文件是否存在
    for key in ["input", "oracle_h5", "gene_list"]:
        if not os.path.exists(args[key]):
            raise FileNotFoundError(f"[ERROR] {key} file not found: {args[key]}\nHint: Check if preprocess/GRN function has run successfully.")

    # 支持多个 multiplier
    multiplier_list = [m.strip() for m in multiplier.split(",") if m.strip()]
    print(f"[INFO] Running multipliers: {multiplier_list}")

    for m in multiplier_list:
        run_for_one_multiplier(args, m, label_key)

    print("\n[INFO] All multipliers completed.")
