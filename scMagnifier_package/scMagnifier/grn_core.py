#!/usr/bin/env python3
"""
grn.py

Full CellOracle GRN construction pipeline:
 1) Load preprocessed AnnData
 2) Extract TF genes from base GRN
 3) Intersect with HVGs → save TF_candidates.txt
 4) Create Oracle object
 5) PCA + KNN imputation
 6) Save initial Oracle
 7) Build Links, filter, compute network score
 8) Build cluster-specific TF dict
 9) Fit GRN for simulation (bagging_ridge)
 10) Save final Oracle

Package usage:
    from scMagnifier import GRN
    
    # 基础使用（默认读取preprocess的输出文件）
    GRN()
    
    # 自定义参数使用
    GRN(
        adata_path="/path/to/your/preprocessed.h5ad",
        outdir="my_GRN_output",
        cluster_key="louvain",
        n_jobs=8
    )
"""
import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import matplotlib.pyplot as plt
import scanpy as sc
import pandas as pd
import celloracle as co
from pathlib import Path  # 统一路径处理

# 设置绘图参数
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["figure.figsize"] = [6, 4.5]


def GRN(
    adata_path: str = os.path.join("preprocessed_result", "preprocessed.h5ad"),  # 默认关联preprocess输出
    base_grn: str | None = None,  # 可选：自定义base GRN路径，默认None（使用内置人类promoter GRN）
    outdir: str = "GRN",  # 可选：输出目录，默认自动创建
    cluster_key: str = "leiden",  # 可选：聚类列名，默认"leiden"
    embedding_name: str = "X_umap",  # 可选：嵌入名称，默认"X_umap"
    oracle_initial_name: str = "preadata.initial.celloracle.oracle",  
    oracle_final_name: str = "preadata.final.celloracle.oracle",  
    links_name: str = "preadata_links.celloracle.links",  
    filter_p: float = 0.001,  # 可选：过滤p值，默认0.001
    threshold_links: int = 2000,  # 可选：链接阈值，默认2000
    n_jobs: int = 4,  # 可选：并行任务数，默认4
    alpha: float = 10.0,  # 可选：alpha参数，默认10.0
    bagging_number: int = 200,  # 可选：bagging数量，默认200
    seed: int = 42  # 可选：随机种子，默认42
) -> None:
    """
    CellOracle GRN构建核心函数（适配preprocess输出）
    
    参数说明：
    ----------
    adata_path : str, optional
        输入预处理后的h5ad文件路径（默认：preprocessed_result/preprocessed.h5ad，即preprocess函数默认输出）
    base_grn : str | None, optional
        自定义base GRN文件路径，默认None（使用CellOracle内置的人类promoter base GRN）
    outdir : str, optional
        GRN结果输出目录（默认：GRN），会自动创建
    cluster_key : str, optional
        adata.obs中聚类结果的列名（默认：leiden）
    embedding_name : str, optional
        adata.obsm中的嵌入矩阵名称（默认：X_umap）
    oracle_initial_name : str, optional
        初始Oracle对象保存名称（默认：preadata.initial.celloracle.oracle）
    oracle_final_name : str, optional
        最终Oracle对象保存名称（默认：preadata.final.celloracle.oracle）
    links_name : str, optional
        Links对象保存名称（默认：preadata_links.celloracle.links）
    filter_p : float, optional
        Links过滤的p值阈值（默认：0.001）
    threshold_links : int, optional
        Links过滤的链接数量阈值（默认：2000）
    n_jobs : int, optional
        并行计算的CPU核心数（默认：4）
    alpha : float, optional
        GRN拟合的alpha正则化参数（默认：10.0）
    bagging_number : int, optional
        bagging_ridge模型的bagging次数（默认：200）
    seed : int, optional
        随机种子（默认：42）
    
    返回值：
    ----------
    None
        结果保存到指定输出目录，包含TF交集列表、Oracle对象、Links对象等
    """
    # 固定随机种子
    np.random.seed(seed)
    
    # 路径处理：转换为Path对象，自动创建输出目录
    adata_path = Path(adata_path)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory prepared: {outdir.absolute()}")

    # 校验输入文件是否存在
    if not adata_path.exists():
        raise FileNotFoundError(f"Input h5ad file not found: {adata_path.absolute()}\nHint: Check if preprocess function has run successfully.")

    # -----------------------
    # Load preprocessed AnnData
    print("Loading preprocessed AnnData from:", adata_path.absolute())
    adata = sc.read_h5ad(adata_path)
    print(f"AnnData shape: {adata.shape}")

    # Use raw_count layer if exists
    if "raw_count" in adata.layers:
        adata.X = adata.layers["raw_count"].copy()
        print("Using adata.layers['raw_count'] as adata.X")

    # -----------------------
    # Determine base GRN
    if base_grn is None or (isinstance(base_grn, str) and base_grn.lower() == "none"):
        print("Using built-in human promoter base GRN (CellOracle default)")
        base_grn = co.data.load_human_promoter_base_GRN()
    else:
        print("Using custom base GRN from:", base_grn)

    # -----------------------
    # Step 1: Extract TFs from base GRN
    print("Extracting TF genes from base GRN ...")
    tf_genes = base_grn.columns[2:].tolist()
    print(f"Total TF genes detected from base GRN: {len(tf_genes)}")

    # -----------------------
    # Step 2: Identify HVGs
    if "highly_variable" not in adata.var.columns:
        print("No 'highly_variable' column found — recomputing HVGs (seurat_v3, n_top_genes=2000) ...")
        sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=2000)
    hvg_genes = adata.var_names[adata.var["highly_variable"]].tolist()
    print(f"Total highly variable genes (HVGs) detected in AnnData: {len(hvg_genes)}")

    # -----------------------
    # Step 3: TF ∩ HVG
    tf_hvg_intersect = sorted(list(set(tf_genes).intersection(set(hvg_genes))))
    print(f"Number of genes in TF-HVG intersection: {len(tf_hvg_intersect)}")

    # Save to txt
    tf_out_path = outdir / "TF_HVG_intersect_genes.txt"
    pd.Series(tf_hvg_intersect).to_csv(tf_out_path, index=False, header=False)
    print(f"TF-HVG intersection gene list saved to: {tf_out_path.absolute()}")

    # -----------------------
    # Create Oracle object
    print("Creating Oracle object ...")
    oracle = co.Oracle()

    print("Importing AnnData into Oracle (raw count) ...")
    oracle.import_anndata_as_raw_count(
        adata=adata,
        cluster_column_name=cluster_key,
        embedding_name=embedding_name
    )

    print("Importing TF data into Oracle ...")
    oracle.import_TF_data(TF_info_matrix=base_grn)

    # -----------------------
    # PCA
    print("Performing PCA on Oracle object ...")
    oracle.perform_PCA()
    plt.plot(np.cumsum(oracle.pca.explained_variance_ratio_)[:100])
    try:
        n_comps = np.where(np.diff(np.diff(np.cumsum(oracle.pca.explained_variance_ratio_))>0.002))[0][0]
    except Exception:
        n_comps = min(50, oracle.pca.n_components_)
    plt.axvline(n_comps, c="k")
    pca_fig_path = outdir / "pca_variance_explained.png"
    plt.savefig(pca_fig_path, bbox_inches="tight")
    plt.close()
    print(f"PCA variance explained figure saved to: {pca_fig_path.absolute()}")
    n_comps = min(n_comps, 50)

    # -----------------------
    # KNN imputation
    n_cell = oracle.adata.shape[0]
    k = max(1, int(0.025 * n_cell))
    print(f"Running KNN imputation with n_pca_dims={n_comps}, k={k}")
    oracle.knn_imputation(
        n_pca_dims=n_comps,
        k=k,
        balanced=True,
        b_sight=k * 8,
        b_maxl=k * 4,
        n_jobs=n_jobs
    )

    # -----------------------
    # Save initial Oracle
    oracle_initial_out = outdir / oracle_initial_name
    print(f"Saving initial Oracle (before GRN fitting) to: {oracle_initial_out.absolute()}")
    oracle.to_hdf5(str(oracle_initial_out)) 

    # -----------------------
    # Build Links
    print("Building Links object from Oracle ...")
    links = oracle.get_links(cluster_name_for_GRN_unit=cluster_key, alpha=alpha, verbose_level=10)
    links.filter_links(p=filter_p, weight="coef_abs", threshold_number=threshold_links)
    links.get_network_score()

    links_out = outdir / links_name
    links.to_hdf5(file_path=str(links_out))
    print(f"Links object saved to: {links_out.absolute()}")

    # -----------------------
    # Build cluster-specific TF dict
    print("Building cluster-specific TF dictionary from Links ...")
    oracle.get_cluster_specific_TFdict_from_Links(links_object=links)

    # -----------------------
    # Fit GRN for simulation
    print("Fitting GRN for simulation (bagging_ridge) ... (this may take time)")
    try:
        oracle.fit_GRN_for_simulation(
            alpha=alpha,
            use_cluster_specific_TFdict=True,
            n_jobs=n_jobs,
            model_method="bagging_ridge",
            bagging_number=bagging_number,
        )
    except Exception as e:
        print(f"Primary oracle.fit_GRN_for_simulation failed (trying fallback): {str(e)}")
        try:
            oracle.fit_GRN_for_simulation(alpha=alpha, use_cluster_specific_TFdict=True)
        except Exception as e2:
            print(f"Fallback fit_GRN_for_simulation also failed: {str(e2)}")
            raise RuntimeError("fit_GRN_for_simulation unavailable for this CellOracle version.") from e2

    # -----------------------
    # Save final Oracle
    oracle_final_out = outdir / oracle_final_name
    print(f"Saving final Oracle with fitted GRN to: {oracle_final_out.absolute()}")
    oracle.to_hdf5(str(oracle_final_out))

    print("="*60)
    print("GRN construction pipeline finished successfully!")
    print(f"All results saved in: {outdir.absolute()}")

