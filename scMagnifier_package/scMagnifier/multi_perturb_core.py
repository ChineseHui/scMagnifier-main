#!/usr/bin/env python3
import warnings
warnings.filterwarnings("ignore")
import os
import gc
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
import celloracle as co
import pandas as pd

try:
    import harmonypy
except ImportError:
    harmonypy = None

try:
    from scanorama import integrate_scanpy
except ImportError:
    integrate_scanpy = None

try:
    from scvi.model import SCVI
    import torch
except ImportError:
    SCVI = None
    torch = None

# -----------------------------
# Helper: 提取GRN系数矩阵
# -----------------------------
def extract_coef_dict(oracle):
    for attr in ["coef_matrix_per_cluster", "coef_dict", "coefs", "coef_mtx_dict"]:
        if hasattr(oracle, attr):
            coef_dict = getattr(oracle, attr)
            print(f"[INFO] Found coefficient matrix in oracle.{attr}")
            return coef_dict
    raise AttributeError("Cannot find coef matrix in oracle object.")

# -----------------------------
# Helper: 扰动传播
# -----------------------------
def propagate_matrix(deltaX, coef_mat, pert_indices, n_iters=3):
    v = deltaX.copy()
    for _ in range(n_iters):
        v_next = v @ coef_mat
        v_next[:, pert_indices] = deltaX[:, pert_indices]  # 保持扰动基因不变
        v = v_next
    return v

# -----------------------------
# Core: 单multiplier的扰动逻辑
# -----------------------------
def run_for_one_multiplier(
    args, 
    multiplier, 
    method: str = "harmony"  # 新增：算法分支参数
):
    # 原代码：生成multiplier目录
    multiplier_str = str(multiplier).replace('.', 'p').replace('-', 'neg')
    outdir = os.path.join(args.outdir, multiplier_str)
    os.makedirs(outdir, exist_ok=True)

    print(f"\n[INFO] === Running for multiplier={multiplier} (method={method}) ===")
    print(f"[INFO] Results will be saved under: {outdir}")

    # Step 1. 加载数据
    adata = sc.read_h5ad(args.input)
    oracle = co.load_hdf5(args.oracle_h5)
    coef_dict = extract_coef_dict(oracle)

    # 使用 raw_count 层作为原始计数矩阵
    if "raw_count" in adata.layers:
        X_raw = adata.layers["raw_count"].copy()
    else:
        X_raw = adata.X.copy()
        print("[WARN] adata.layers['raw_count'] not found; using adata.X as raw counts.")

    gene_names = list(adata.var_names)
    cluster_labels = adata.obs[args.cluster_key].astype(str)
    clusters = sorted(cluster_labels.unique())
    print(f"[INFO] Found {len(clusters)} clusters in {args.cluster_key}: {clusters}")

    # 保存原始矩阵
    raw_csv_path = os.path.join(outdir, "original_matrix.csv")
    pd.DataFrame(X_raw, columns=gene_names, index=adata.obs_names).to_csv(raw_csv_path)
    print(f"[SAVED] Original expression matrix: {raw_csv_path}")

    old_umap = adata.obsm["X_umap"].copy() if "X_umap" in adata.obsm else None

    # Step 2. 读取扰动基因列表
    with open(args.gene_list, "r") as f:
        genes_to_perturb = [line.strip() for line in f if line.strip()]
    print(f"[INFO] Loaded {len(genes_to_perturb)} genes from txt file.")

    multiplier = float(multiplier)

    # Step 3. 遍历基因扰动
    for gene in genes_to_perturb:
        print("=" * 80)
        print(f"[RUNNING] Perturbing gene: {gene}  (multiplier={multiplier})")

        if gene not in gene_names:
            print(f"[WARN] Gene {gene} not found in adata.var_names, skipping.")
            continue

        pert_idx = gene_names.index(gene)
        X_new_total = X_raw.copy()

        # 扰动传播
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

        # 保存扰动矩阵
        perturbed_csv_path = os.path.join(outdir, f"perturbed_matrix_{gene}_{multiplier}.csv")
        pd.DataFrame(X_new_total, columns=gene_names, index=adata.obs_names).to_csv(perturbed_csv_path)
        print(f"[SAVED] Perturbed matrix for {gene}: {perturbed_csv_path}")

        # -----------------------------
        # 核心差异：根据method分支处理下游分析
        # -----------------------------
        adata_tmp = adata.copy()
        pert_cluster_key = None
        use_rep = None

        if method == "harmony":
            # ====== Harmony分支======
            adata_tmp.X = X_new_total.copy()
            print("[INFO] Running log1p + scale + PCA + Harmony + neighbors + UMAP + Clustering ...")
            
            # 预处理
            sc.pp.log1p(adata_tmp)
            sc.pp.scale(adata_tmp, max_value=10)
            sc.tl.pca(adata_tmp, svd_solver="arpack", n_comps=args.n_pcs)

            # Harmony批次校正
            if harmonypy is None:
                raise ImportError("使用Harmony需要安装harmonypy！\n安装命令：pip install harmonypy>=0.0.9")
            if args.batch_key not in adata_tmp.obs.columns:
                raise ValueError(f"ERROR: batch_key '{args.batch_key}' not found in adata.obs!")
            
            sc.external.pp.harmony_integrate(
                adata_tmp,
                key=args.batch_key,                
                basis='X_pca',
                adjusted_basis='X_pca_harmony',
                random_state=args.seed
            )
            print("[INFO] Harmony batch correction completed. Proceeding with neighbor graph...")
            use_rep = "X_pca_harmony"

        elif method == "scanorama":
            # ====== Scanorama分支======
            adata_tmp.X = X_new_total.copy()
            print("[INFO] Running log1p + scale + PCA + Scanorama + neighbors + UMAP + Clustering ...")
            
            # 预处理
            sc.pp.log1p(adata_tmp)
            sc.pp.scale(adata_tmp, max_value=10)

            # Scanorama批次校正
            if integrate_scanpy is None:
                raise ImportError("使用Scanorama需要安装scanorama！\n安装命令：pip install scanorama>=1.7.0")
            if args.batch_key not in adata_tmp.obs.columns:
                raise ValueError(f"ERROR: batch_key='{args.batch_key}' not found in adata.obs!")
            
            batches = adata_tmp.obs[args.batch_key].unique().tolist()
            adatas = [adata_tmp[adata_tmp.obs[args.batch_key] == b].copy() for b in batches]
            try:
                integrate_scanpy(adatas)
            except Exception as e:
                raise RuntimeError("Scanorama integration failed: " + str(e))

            for ad in adatas:
                if 'X_scanorama' not in ad.obsm:
                    ad.obsm['X_scanorama'] = ad.X.copy()
            adata_tmp = adatas[0].concatenate(*adatas[1:], batch_key="batch_sc", index_unique=None)

            if 'X_scanorama' not in adata_tmp.obsm:
                try:
                    X_scanorama_list = [ad.obsm['X_scanorama'] for ad in adatas]
                    adata_tmp.obsm['X_scanorama'] = np.vstack(X_scanorama_list)
                except Exception:
                    adata_tmp.obsm['X_scanorama'] = adata_tmp.X.copy()
            print("[INFO] Scanorama integration complete. Corrected embedding saved in adata_tmp.obsm['X_scanorama'].")
            use_rep = "X_scanorama"

        elif method == "scvi":
            # ====== scVI分支======
            adata_tmp.layers["raw_count"] = X_new_total.copy()
            print(f"[INFO] Running scVI batch correction on perturbed matrix (max_epochs={args.max_epochs})...")

            if SCVI is None or torch is None:
                raise ImportError("使用scVI需要安装scvi-tools和torch！\n安装命令：pip install scvi-tools>=1.0.3 torch>=2.0.1")
            if args.batch_key not in adata_tmp.obs.columns:
                raise ValueError(f"batch_key '{args.batch_key}' not found in adata.obs!")
            
            # SCVI训练
            SCVI.setup_anndata(adata_tmp, layer="raw_count", batch_key=args.batch_key)
            model = SCVI(adata_tmp, n_latent=args.n_pcs, gene_likelihood="nb")
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model.train(
                max_epochs=args.max_epochs, 
                accelerator=device,
                enable_progress_bar=True
            )
            latent = model.get_latent_representation()
            adata_tmp.obsm["X_scVI"] = latent
            use_rep = "X_scVI"

        else:
            raise ValueError(f"不支持的method：{method}，仅支持harmony/scanorama/scvi")

        # -----------------------------
        # 公共下游步骤：neighbors + UMAP + 聚类
        # -----------------------------
        # Neighbors
        sc.pp.neighbors(adata_tmp, use_rep=use_rep, n_neighbors=args.n_neighbors, random_state=args.seed)
        # UMAP
        sc.tl.umap(adata_tmp, random_state=args.seed)
        # 聚类
        if args.cluster_key.lower() == "leiden":
            sc.tl.leiden(adata_tmp, resolution=args.resolution,
                         key_added="leiden_perturbed", random_state=args.seed)
            pert_cluster_key = "leiden_perturbed"
        elif args.cluster_key.lower() == "louvain":
            sc.tl.louvain(adata_tmp, resolution=args.resolution,
                          key_added="louvain_perturbed", random_state=args.seed)
            pert_cluster_key = "louvain_perturbed"
        else:
            raise ValueError("cluster_key must be either 'leiden' or 'louvain'")

        adata_tmp.obsm["X_umap_perturbed"] = adata_tmp.obsm["X_umap"].copy()

        # 保存聚类结果
        cluster_csv = os.path.join(outdir, f"cluster_{gene}_{multiplier}.csv")
        adata_tmp.obs[[pert_cluster_key]].to_csv(cluster_csv)
        print(f"[SAVED] {cluster_csv}")

        # -----------------------------
        # UMAP绘图
        # -----------------------------
        if args.plot_umap:
            # 聚类结果UMAP
            try:
                fig1 = sc.pl.umap(
                    adata_tmp,
                    color=pert_cluster_key,
                    size=8,
                    alpha=0.8,
                    show=False,
                    return_fig=True
                )
                fig1.savefig(
                    os.path.join(outdir, f"umap_new_{pert_cluster_key}_{gene}_{multiplier}.png"),
                    bbox_inches="tight"
                )
                plt.close(fig1)
            except Exception as e:
                print(f"[ERROR] Failed to plot UMAP for {pert_cluster_key}: {e}")

            # 标签UMAP
            if args.label_key in adata_tmp.obs.columns:
                try:
                    fig2 = sc.pl.umap(
                        adata_tmp,
                        color=args.label_key,      
                        size=8,
                        alpha=0.8,
                        show=False,
                        return_fig=True
                    )
                    fig2.savefig(
                        os.path.join(outdir, f"umap_label_{gene}_{multiplier}.png"),
                        bbox_inches="tight"
                    )
                    plt.close(fig2)
                except Exception as e:
                    print(f"[ERROR] Failed to plot label_key ({args.label_key}): {e}")

            # 原始UMAP投影
            if old_umap is not None:
                try:
                    adata_tmp.obsm["X_umap"] = old_umap.copy()
                    adata_tmp.obs["pert_cluster_str"] = adata_tmp.obs[pert_cluster_key].astype(str)
                    fig3 = sc.pl.umap(
                        adata_tmp,
                        color="pert_cluster_str",
                        size=8,
                        alpha=0.8,
                        show=False,
                        return_fig=True
                    )
                    fig3.savefig(
                        os.path.join(outdir, f"umap_original_on_new_{gene}_{multiplier}.png"),
                        bbox_inches="tight"
                    )
                    plt.close(fig3)
                    adata_tmp.obsm["X_umap"] = adata_tmp.obsm["X_umap_perturbed"].copy()
                except Exception as e:
                    print(f"[ERROR] Failed to project old UMAP: {e}")

        # -----------------------------
        # scVI特有：内存清理
        # -----------------------------
        if method == "scvi":
            del adata_tmp, model, latent
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        else:
            del adata_tmp

        print(f"[DONE] Perturbation for {gene} completed.")
        print("=" * 80)

    print(f"\n[INFO] All genes done for multiplier={multiplier}.")

# -----------------------------
# 主函数：multi_perturb
# -----------------------------
def multi_perturb(
    method: str = "harmony",  # 可选：批次校正算法（默认harmony）
    input: str = os.path.join("preprocessed_result", "preprocessed.h5ad"),  # 按要求设置默认值
    oracle_h5: str = os.path.join("GRN", "preadata.final.celloracle.oracle"),  # 按要求设置默认值
    gene_list: str = os.path.join("GRN", "TF_HVG_intersect_genes.txt"),  # 按要求设置默认值
    multiplier: str = "0.1,-0.1",  # 可选：扰动系数列表
    n_pcs: int = 20,  # 可选：PCA/scVI潜在维度
    n_neighbors: int = 10,  # 可选：近邻数
    resolution: float = 0.75,  # 可选：聚类分辨率
    seed: int = 42,  # 可选：随机种子
    batch_key: str = "batch",  # 可选：批次列名
    cluster_key: str = "leiden",  # 可选：聚类方法（leiden/louvain）
    label_key: str = "assigned_cluster",  # 可选：标签列名
    outdir: str = "perturb_results",  # 可选：输出目录
    plot_umap: bool = True,  # 可选：是否绘制UMAP
    save_h5ad: bool = False,  # 可选：是否保存h5ad
    max_epochs: int = 100,  # 新增：scVI训练最大轮数（仅scVI生效，默认100）
) -> None:
    """
    多算法整合的单细胞基因扰动分析函数（支持Harmony/Scanorama/scVI批次校正）
    核心逻辑与原三个独立代码完全一致，仅通过method参数切换批次校正算法。

    参数说明：
    ----------
    method : str, optional
        批次校正算法，可选值：harmony/scanorama/scvi，默认值："harmony"
    input : str, optional
        输入预处理后的h5ad文件路径，默认值："preprocessed_result/preprocessed.h5ad"
    oracle_h5 : str, optional
        CellOracle的GRN模型文件路径，默认值："GRN/preadata.final.celloracle.oracle"
    gene_list : str, optional
        扰动基因列表txt文件路径，默认值："GRN/TF_HVG_intersect_genes.txt"
    multiplier : str, optional
        扰动系数列表（逗号分隔），默认值："0.1,-0.1"
    n_pcs : int, optional
        PCA/scVI潜在维度，默认值：20
    n_neighbors : int, optional
        近邻数，默认值：10
    resolution : float, optional
        聚类分辨率，默认值：0.75
    seed : int, optional
        随机种子，默认值：42
    batch_key : str, optional
        adata.obs中批次列名，默认值："batch"
    cluster_key : str, optional
        聚类方法（leiden/louvain），默认值："leiden"
    label_key : str, optional
        adata.obs中细胞标签列名，默认值："assigned_cluster"
    outdir : str, optional
        输出目录，默认值："perturb_result"
    plot_umap : bool, optional
        是否绘制UMAP图，默认值：True
    save_h5ad : bool, optional
        是否保存扰动后的h5ad文件，默认值：False
    max_epochs : int, optional
        scVI模型训练的最大轮数（仅对scVI生效），默认值：100
    """
    # 参数预处理
    method = method.lower()
    valid_methods = ["harmony", "scanorama", "scvi"]
    if method not in valid_methods:
        raise ValueError(f"method必须是{valid_methods}之一，当前输入：{method}")
    
    # 校验max_epochs
    if method == "scvi" and max_epochs <= 0:
        raise ValueError(f"scVI的max_epochs必须为正整数，当前输入：{max_epochs}")
    
    # 构建参数对象
    class Args:
        pass
    args = Args()
    args.input = input
    args.oracle_h5 = oracle_h5
    args.gene_list = gene_list
    args.n_pcs = n_pcs
    args.n_neighbors = n_neighbors
    args.resolution = resolution
    args.seed = seed
    args.batch_key = batch_key
    args.cluster_key = cluster_key
    args.label_key = label_key
    args.outdir = outdir
    args.plot_umap = plot_umap
    args.save_h5ad = save_h5ad
    args.max_epochs = max_epochs  # 新增：传入max_epochs

    # 解析扰动系数列表
    multiplier_list = [m.strip() for m in multiplier.split(",") if m.strip()]
    if not multiplier_list:
        raise ValueError("multiplier列表不能为空！示例：'0.1,-0.1'")
    print(f"[INFO] Running multipliers: {multiplier_list} (method={method})")
    if method == "scvi":
        print(f"[INFO] scVI training config: max_epochs={args.max_epochs}")

    # 遍历每个扰动系数执行分析
    for m in multiplier_list:
        run_for_one_multiplier(args, m, method=method)

    print("\n[INFO] All multipliers completed.")
