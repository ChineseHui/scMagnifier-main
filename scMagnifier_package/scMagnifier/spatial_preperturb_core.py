#!/usr/bin/env python3
"""
scMagnifier/spatial_preperturb.py
功能：读取celloracle oracle文件，提取每个cluster的GRN系数矩阵，保存为npz格式
"""
import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import celloracle as co
from typing import Dict, Any, Optional

# ========================================================
#  核心函数：spatial_preperturb
# ========================================================
def spatial_preperturb(
    # 关键修正：默认路径匹配GRN函数的默认输出（outdir="GRN" + oracle_final_name="preadata.final.celloracle.oracle"）
    oracle_path: str = os.path.join("GRN", "preadata.final.celloracle.oracle"),
    outdir: str = "GRN"
) -> str:
    """
    提取celloracle oracle文件中的GRN系数矩阵，保存为压缩的npz格式

    Parameters
    ----------
    oracle_path : str, optional
        输入的celloracle oracle文件路径，默认匹配GRN函数的默认输出路径：GRN/preadata.final.celloracle.oracle
    outdir : str, optional
        输出结果目录，默认 "GRN"（与GRN函数默认输出目录一致）

    Returns
    -------
    str
        保存的npz文件完整路径，方便后续调用

    Raises
    ------
    FileNotFoundError
        输入的oracle文件路径不存在时抛出
    AttributeError
        oracle对象中未找到GRN系数矩阵时抛出
    """
    # 1. 输入校验
    if not os.path.exists(oracle_path):
        raise FileNotFoundError(
            f"Oracle file not found: {oracle_path}\n"
            "Hint: 1. 先运行GRN函数生成final oracle文件；2. 若GRN函数自定义了outdir/oracle_final_name，请同步修改此处oracle_path"
        )
    
    # 2. 创建输出目录
    os.makedirs(outdir, exist_ok=True)

    # 3. 辅助函数：提取oracle对象中的系数矩阵（嵌套在核心函数内，更内聚）
    def extract_coef_dict(oracle: co.Oracle) -> Dict[Any, np.ndarray]:
        """从oracle对象中提取GRN系数矩阵字典"""
        coef_attrs = ["coef_matrix_per_cluster", "coef_dict", "coefs", "coef_mtx_dict"]
        for attr in coef_attrs:
            if hasattr(oracle, attr):
                coef_dict = getattr(oracle, attr)
                print(f"[INFO] Found coefficient matrix in oracle.{attr}")
                return coef_dict
        raise AttributeError("Cannot find coefficient matrix in oracle object.")

    # 4. 加载oracle文件
    print("[INFO] Loading oracle file ...")
    oracle = co.load_hdf5(oracle_path)

    # 5. 提取系数矩阵并转换为纯numpy格式
    coef_dict = extract_coef_dict(oracle)
    coef_np = {}
    for cid, mat in coef_dict.items():
        coef_np[str(cid)] = np.asarray(mat, dtype=np.float32)  

    # 6. 保存为压缩的npz文件
    save_path = os.path.join(outdir, "celloracle_grn_coef.npz")
    np.savez_compressed(save_path, **coef_np)

    # 7. 输出提示并返回保存路径
    print(f"[SAVED] GRN coefficients saved to {save_path}")
    print("[DONE] Spatial preperturb process completed")

