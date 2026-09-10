import os
import sys
import re
import random
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import scipy.sparse as sp

from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

project_root = "/data2/liangyefeng/My_GraphST_Innovation"
sys.path.insert(0, os.path.join(project_root, "src"))

from SpaGCN.SpaGCN_package.SpaGCN.SpaGCN import SpaGCN
from SpaGCN.SpaGCN_package.SpaGCN.util import (
    prefilter_genes, prefilter_specialgenes, search_l, search_res
)
from SpaGCN.SpaGCN_package.SpaGCN.calculate_adj import calculate_adj_matrix


# =========================
# Config
# =========================
DATA_PATHS = [
    "/data2/liangyefeng/My_GraphST_Innovation/data/MERFISH/fbe9c55e-518b-4bb7-9e1d-00322be93695.h5ad",
    "/data2/liangyefeng/My_GraphST_Innovation/data/MERFISH/bf8cb800-86b2-4ac0-b8a7-cc89fedaec89.h5ad",
]

OUT_DIR = "/data2/liangyefeng/My_GraphST_Innovation/data/MERFISH/SpaGCN_binned_raw_results"
os.makedirs(OUT_DIR, exist_ok=True)

RESULT_CSV = os.path.join(OUT_DIR, "SpaGCN_MERFISH_binned_raw_results.csv")

SEED = 41

FORCE_LABEL_COL = "cell_type"
FORCE_SLICE_COL = "slice"

FORCE_SPATIAL_KEY = None
FORCE_X_COL = None
FORCE_Y_COL = None

# ===== Binning params =====
# SpaGCN adjacency 矩阵是全矩阵，bin 不要设太小，先用 100 稳一点
BIN_SIZE = 100.0
MIN_CELLS_PER_BIN = 3

# ===== SpaGCN params =====
SEARCH_L_P = 0.5
SEARCH_L_START = 0.01
SEARCH_L_END = 1000
SEARCH_L_TOL = 0.01
SEARCH_L_MAX_RUN = 100

SEARCH_RES_START = 0.7
SEARCH_RES_STEP = 0.1
SEARCH_RES_TOL = 5e-3
SEARCH_RES_LR = 0.05
SEARCH_RES_MAX_EPOCHS = 20
SEARCH_RES_MAX_RUN = 20

TRAIN_INIT_SPA = True
TRAIN_INIT = "louvain"
TRAIN_TOL = 5e-3
TRAIN_LR = 0.05
TRAIN_MAX_EPOCHS = 200

MIN_GENES_PER_GENE = 3
MIN_COUNTS_PER_BIN = 1
MIN_OBS_PER_SECTION = 50

INVALID_LABELS = {
    "", "na", "n/a", "nan", "none", "null",
    "unknown", "unassigned", "unlabeled", "unlabelled", "undefined"
}


# =========================
# Utils
# =========================
def seed_everything(seed=41):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def safe_name(x):
    x = str(x)
    x = re.sub(r"[^\w\-.]+", "_", x)
    return x[:200]


def print_header(msg):
    print("\n" + "=" * 120)
    print(msg)
    print("=" * 120)


def make_obs_names_unique(adata):
    if not adata.obs_names.is_unique:
        adata.obs_names_make_unique()
    return adata


def ensure_spatial(adata, force_spatial_key=None, force_x_col=None, force_y_col=None):
    if force_spatial_key is not None:
        if force_spatial_key not in adata.obsm_keys():
            raise ValueError(f"FORCE_SPATIAL_KEY='{force_spatial_key}' 不在 adata.obsm 中")
        arr = np.asarray(adata.obsm[force_spatial_key])
        if arr.ndim != 2 or arr.shape[1] < 2:
            raise ValueError(f"adata.obsm['{force_spatial_key}'] 不是合法二维坐标")
        adata.obsm["spatial"] = arr[:, :2].astype(np.float64)
        return "obsm:" + force_spatial_key

    # 优先用更合适的局部坐标
    for key in ["X_spatial_coords", "spatial_coords", "spatial", "X_spatial", "coordinates", "coord"]:
        if key in adata.obsm_keys():
            arr = np.asarray(adata.obsm[key])
            if arr.ndim == 2 and arr.shape[1] >= 2:
                adata.obsm["spatial"] = arr[:, :2].astype(np.float64)
                return "obsm:" + key

    obs = adata.obs

    if force_x_col is not None and force_y_col is not None:
        if force_x_col not in obs.columns or force_y_col not in obs.columns:
            raise ValueError("FORCE_X_COL / FORCE_Y_COL 不在 adata.obs.columns 中")
        coords = np.c_[obs[force_x_col].values, obs[force_y_col].values].astype(np.float64)
        adata.obsm["spatial"] = coords
        return f"obs:{force_x_col},{force_y_col}"

    possible_pairs = [
        ("center_x", "center_y"),
        ("x", "y"),
        ("X", "Y"),
        ("x_centroid", "y_centroid"),
        ("x_center", "y_center"),
        ("row", "col"),
        ("array_row", "array_col"),
        ("pxl_row_in_fullres", "pxl_col_in_fullres"),
    ]

    for x_col, y_col in possible_pairs:
        if x_col in obs.columns and y_col in obs.columns:
            coords = np.c_[obs[x_col].values, obs[y_col].values].astype(np.float64)
            adata.obsm["spatial"] = coords
            return f"obs:{x_col},{y_col}"

    raise ValueError("没有找到空间坐标")


def summarize_matrix_quick(X, tag):
    if sp.issparse(X):
        data = X.data
        if data.size == 0:
            print(f"[{tag}] sparse empty")
            return
        print(
            f"[{tag}] sparse shape={X.shape}, nnz={X.nnz}, "
            f"min={float(np.min(data))}, max={float(np.max(data))}, "
            f"nan={int(np.isnan(data).sum())}, inf={int(np.isinf(data).sum())}, neg={int((data < 0).sum())}"
        )
    else:
        arr = np.asarray(X, dtype=np.float64)
        print(
            f"[{tag}] dense shape={arr.shape}, "
            f"min={float(np.nanmin(arr))}, max={float(np.nanmax(arr))}, "
            f"nan={int(np.isnan(arr).sum())}, inf={int(np.isinf(arr).sum())}, neg={int((arr < 0).sum())}"
        )


def check_matrix_nonnegative(X, tag):
    if sp.issparse(X):
        data = X.data
        if np.isnan(data).sum() > 0 or np.isinf(data).sum() > 0:
            raise ValueError(f"{tag}: 存在 NaN/Inf")
        if data.size > 0 and np.min(data) < 0:
            raise ValueError(f"{tag}: 存在负值，不能作为 counts 使用")
    else:
        arr = np.asarray(X, dtype=np.float64)
        if np.isnan(arr).sum() > 0 or np.isinf(arr).sum() > 0:
            raise ValueError(f"{tag}: 存在 NaN/Inf")
        if np.min(arr) < 0:
            raise ValueError(f"{tag}: 存在负值，不能作为 counts 使用")


def clean_label_series(series):
    s = series.astype("string").copy()
    s = s.str.strip()
    valid = ~s.isna()
    valid &= ~s.str.lower().isin(INVALID_LABELS)
    return s, np.asarray(valid, dtype=bool)


def compute_ari_nmi_from_arrays(pred, gt):
    pred = pd.Series(pred).astype(str)
    gt = pd.Series(gt).astype(str)

    valid = (~pred.isna()) & (~gt.isna())
    pred = pred[valid]
    gt = gt[valid]

    if len(pred) == 0:
        raise ValueError("没有可用于评估的样本")
    if gt.nunique() < 2:
        raise ValueError("GT 类别数 < 2，无法计算 ARI/NMI")

    ari = adjusted_rand_score(gt, pred)
    nmi = normalized_mutual_info_score(gt, pred, average_method="arithmetic")
    return float(ari), float(nmi), int(len(pred))


def get_sections(adata, slice_col=None):
    if slice_col is None:
        return [("all", np.arange(adata.n_obs, dtype=int))]

    values = adata.obs[slice_col].astype(str).to_numpy()
    sections = []

    for v in pd.unique(values):
        if str(v).lower() in {"nan", "none", "na", ""}:
            continue
        idx = np.where(values == str(v))[0]
        if len(idx) > 0:
            sections.append((str(v), idx))

    if len(sections) == 0:
        return [("all", np.arange(adata.n_obs, dtype=int))]

    return sections


def build_binned_adata_from_raw(adata, label_col, bin_size=100.0, min_cells_per_bin=3):
    adata = adata.copy()
    adata = make_obs_names_unique(adata)
    ensure_spatial(adata, FORCE_SPATIAL_KEY, FORCE_X_COL, FORCE_Y_COL)

    if adata.raw is None:
        raise ValueError("adata.raw is None，当前脚本要求使用 raw.X 作为原始 counts")

    X_raw = adata.raw.X
    var_raw = adata.raw.var.copy()

    if not sp.issparse(X_raw):
        X_raw = sp.csr_matrix(X_raw)
    else:
        X_raw = X_raw.tocsr()

    check_matrix_nonnegative(X_raw, "raw_counts")

    coords = np.asarray(adata.obsm["spatial"], dtype=np.float64)
    x = coords[:, 0]
    y = coords[:, 1]

    x0 = np.min(x)
    y0 = np.min(y)

    bin_x = np.floor((x - x0) / bin_size).astype(np.int64)
    bin_y = np.floor((y - y0) / bin_size).astype(np.int64)

    mi = pd.MultiIndex.from_arrays([bin_x, bin_y], names=["bin_x", "bin_y"])
    bin_codes, unique_bins = pd.factorize(mi, sort=True)
    n_bins_all = len(unique_bins)
    n_cells = adata.n_obs

    G = sp.csr_matrix(
        (np.ones(n_cells, dtype=np.float32), (bin_codes, np.arange(n_cells))),
        shape=(n_bins_all, n_cells)
    )

    # raw counts 聚合
    X_bin = (G @ X_raw).tocsr()

    n_cells_per_bin = np.bincount(bin_codes, minlength=n_bins_all).astype(int)

    x_sum = np.bincount(bin_codes, weights=x, minlength=n_bins_all)
    y_sum = np.bincount(bin_codes, weights=y, minlength=n_bins_all)
    x_mean = x_sum / np.maximum(n_cells_per_bin, 1)
    y_mean = y_sum / np.maximum(n_cells_per_bin, 1)

    labels_raw, valid_label_mask = clean_label_series(adata.obs[label_col])
    df_lab = pd.DataFrame({
        "bin_code": bin_codes[valid_label_mask],
        "label": labels_raw[valid_label_mask].astype(str).values
    })

    if len(df_lab) == 0:
        raise ValueError("没有可用于构建 bin 标签的有效细胞标签")

    ct = pd.crosstab(df_lab["bin_code"], df_lab["label"])
    majority_label = pd.Series(index=np.arange(n_bins_all), dtype="object")
    majority_count = pd.Series(0, index=np.arange(n_bins_all), dtype="int64")
    valid_label_count = pd.Series(0, index=np.arange(n_bins_all), dtype="int64")

    majority_label.loc[ct.index] = ct.idxmax(axis=1)
    majority_count.loc[ct.index] = ct.max(axis=1).astype(int)
    valid_label_count.loc[ct.index] = ct.sum(axis=1).astype(int)

    purity = majority_count.values / np.maximum(valid_label_count.values, 1)

    keep_bins = (n_cells_per_bin >= min_cells_per_bin) & (valid_label_count.values > 0)
    keep_idx = np.where(keep_bins)[0]

    if len(keep_idx) < 2:
        raise ValueError("binning 后保留下来的 bins 太少")

    X_bin_keep = X_bin[keep_idx].tocsr()
    spatial_keep = np.c_[x_mean[keep_idx], y_mean[keep_idx]]

    obs_bin = pd.DataFrame({
        "bin_x": np.asarray(unique_bins.get_level_values(0))[keep_idx],
        "bin_y": np.asarray(unique_bins.get_level_values(1))[keep_idx],
        "n_cells": n_cells_per_bin[keep_idx],
        "valid_label_count": valid_label_count.values[keep_idx].astype(int),
        "majority_count": majority_count.values[keep_idx].astype(int),
        "purity": purity[keep_idx],
        "majority_label": majority_label.values[keep_idx].astype(str),
    }, index=[f"bin_{i}" for i in range(len(keep_idx))])

    adata_bin = ad.AnnData(X=X_bin_keep, obs=obs_bin, var=var_raw.copy())
    adata_bin.obsm["spatial"] = spatial_keep.astype(np.float64)
    adata_bin = make_obs_names_unique(adata_bin)

    old_to_new = np.full(n_bins_all, -1, dtype=int)
    old_to_new[keep_idx] = np.arange(len(keep_idx), dtype=int)
    cell_to_bin_new = old_to_new[bin_codes]

    return adata_bin, cell_to_bin_new


def preprocess_for_spagcn(adata):
    adata = adata.copy()
    adata = make_obs_names_unique(adata)

    check_matrix_nonnegative(adata.X, "SpaGCN_preprocess_input")

    prefilter_genes(adata, min_cells=MIN_GENES_PER_GENE)
    prefilter_specialgenes(adata)

    if adata.n_obs == 0:
        raise ValueError("prefilter 后 adata.n_obs = 0")
    if adata.n_vars == 0:
        raise ValueError("prefilter 后 adata.n_vars = 0")

    try:
        sc.pp.normalize_per_cell(adata)
    except Exception:
        sc.pp.normalize_total(adata)

    sc.pp.log1p(adata)

    if sp.issparse(adata.X):
        adata.X = adata.X.tocsr()

    return adata


# =========================
# Main
# =========================
def main():
    seed_everything(SEED)
    results = []

    for data_path in DATA_PATHS:
        dataset_name = os.path.basename(data_path).replace(".h5ad", "")
        print_header(f"LOAD DATASET: {dataset_name}")

        try:
            adata_all = sc.read_h5ad(data_path)
            adata_all = make_obs_names_unique(adata_all)

            print(f"[{dataset_name}] n_obs={adata_all.n_obs}, n_vars={adata_all.n_vars}")
            print(f"[{dataset_name}] obs columns={list(adata_all.obs.columns)}")
            print(f"[{dataset_name}] obsm_keys={list(adata_all.obsm_keys())}")
            print(f"[{dataset_name}] raw is None = {adata_all.raw is None}")

            spatial_source = ensure_spatial(
                adata_all,
                force_spatial_key=FORCE_SPATIAL_KEY,
                force_x_col=FORCE_X_COL,
                force_y_col=FORCE_Y_COL
            )
            print(f"[{dataset_name}] spatial source = {spatial_source}")

            if FORCE_LABEL_COL not in adata_all.obs.columns:
                raise ValueError(f"'{FORCE_LABEL_COL}' 不在 obs 中")
            if FORCE_SLICE_COL not in adata_all.obs.columns:
                raise ValueError(f"'{FORCE_SLICE_COL}' 不在 obs 中")

            label_col = FORCE_LABEL_COL
            slice_col = FORCE_SLICE_COL

            sections = get_sections(adata_all, slice_col=slice_col)
            print(f"[{dataset_name}] total sections to run = {len(sections)}")

            for section_name, idx in sections:
                run_name = f"{dataset_name}__{safe_name(section_name)}"
                print_header(f"START SECTION: {run_name}")

                try:
                    seed_everything(SEED)

                    adata_sec = adata_all[idx].copy()
                    adata_sec = make_obs_names_unique(adata_sec)
                    ensure_spatial(adata_sec, FORCE_SPATIAL_KEY, FORCE_X_COL, FORCE_Y_COL)

                    if adata_sec.n_obs < MIN_OBS_PER_SECTION:
                        raise ValueError(f"n_obs={adata_sec.n_obs} < MIN_OBS_PER_SECTION={MIN_OBS_PER_SECTION}")
                    if adata_sec.raw is None:
                        raise ValueError("section adata.raw is None")

                    gt_cells_raw, valid_cells_mask = clean_label_series(adata_sec.obs[label_col])
                    n_clusters = pd.Series(gt_cells_raw[valid_cells_mask]).nunique()

                    if n_clusters < 2:
                        raise ValueError("当前 section 有效类别数 < 2")

                    print(f"[{run_name}] raw n_obs={adata_sec.n_obs}, n_vars={adata_sec.n_vars}, n_clusters={n_clusters}")
                    print(f"[{run_name}] valid GT cells = {int(valid_cells_mask.sum())}")

                    summarize_matrix_quick(adata_sec.X, f"{run_name} section_X_processed")
                    summarize_matrix_quick(adata_sec.raw.X, f"{run_name} section_raw_X")

                    # ===== raw-count binning =====
                    adata_bin, cell_to_bin_new = build_binned_adata_from_raw(
                        adata_sec,
                        label_col=label_col,
                        bin_size=BIN_SIZE,
                        min_cells_per_bin=MIN_CELLS_PER_BIN
                    )

                    print(f"[{run_name}] after raw-count binning: n_bins={adata_bin.n_obs}, n_vars={adata_bin.n_vars}")
                    print(f"[{run_name}] bin purity mean={adata_bin.obs['purity'].astype(float).mean():.4f}")
                    summarize_matrix_quick(adata_bin.X, f"{run_name} binned_raw_counts")

                    # ===== preprocess for SpaGCN =====
                    adata_bin = preprocess_for_spagcn(adata_bin)
                    print(f"[{run_name}] after preprocess: n_bins={adata_bin.n_obs}, n_vars={adata_bin.n_vars}")
                    summarize_matrix_quick(adata_bin.X, f"{run_name} binned_after_preprocess")

                    # ===== adjacency =====
                    x = adata_bin.obsm["spatial"][:, 0].tolist()
                    y = adata_bin.obsm["spatial"][:, 1].tolist()
                    adj = calculate_adj_matrix(x=x, y=y, histology=False)
                    print(f"[{run_name}] adj shape = {np.asarray(adj).shape}")

                    # ===== search l =====
                    seed_everything(SEED)
                    try:
                        l = search_l(
                            SEARCH_L_P, adj,
                            start=SEARCH_L_START, end=SEARCH_L_END,
                            tol=SEARCH_L_TOL, max_run=SEARCH_L_MAX_RUN
                        )
                    except Exception as e:
                        print(f"[{run_name}] search_l failed: {e}")
                        l = None

                    if l is None:
                        l = 1.0

                    # ===== search res =====
                    seed_everything(SEED)
                    try:
                        res = search_res(
                            adata_bin, adj, l,
                            target_num=n_clusters,
                            start=SEARCH_RES_START,
                            step=SEARCH_RES_STEP,
                            tol=SEARCH_RES_TOL,
                            lr=SEARCH_RES_LR,
                            max_epochs=SEARCH_RES_MAX_EPOCHS,
                            r_seed=SEED, t_seed=SEED, n_seed=SEED,
                            max_run=SEARCH_RES_MAX_RUN
                        )
                    except Exception as e:
                        print(f"[{run_name}] search_res failed: {e}")
                        res = None

                    if res is None:
                        res = 1.0

                    print(f"[{run_name}] recommended l = {l}")
                    print(f"[{run_name}] recommended res = {res}")

                    # ===== train =====
                    clf = SpaGCN()
                    clf.set_l(l)

                    seed_everything(SEED)
                    clf.train(
                        adata_bin, adj,
                        init_spa=TRAIN_INIT_SPA,
                        init=TRAIN_INIT,
                        res=res,
                        tol=TRAIN_TOL,
                        lr=TRAIN_LR,
                        max_epochs=TRAIN_MAX_EPOCHS
                    )

                    pred, _ = clf.predict()
                    pred = np.asarray(pred).astype(int)
                    adata_bin.obs["domain"] = pd.Categorical(pred.astype(str))

                    # ===== bin-level metrics =====
                    bin_ari, bin_nmi, n_bin_eval = compute_ari_nmi_from_arrays(
                        pred=adata_bin.obs["domain"].astype(str).values,
                        gt=adata_bin.obs["majority_label"].astype(str).values
                    )

                    # ===== projected cell-level metrics =====
                    valid_cell_eval = (cell_to_bin_new >= 0) & valid_cells_mask
                    if valid_cell_eval.sum() == 0:
                        raise ValueError("没有可用于 projected cell-level 评估的细胞")

                    bin_domain = adata_bin.obs["domain"].astype(str).values
                    cell_pred = bin_domain[cell_to_bin_new[valid_cell_eval]]
                    cell_gt = gt_cells_raw[valid_cell_eval].astype(str).values

                    cell_ari, cell_nmi, n_cell_eval = compute_ari_nmi_from_arrays(
                        pred=cell_pred,
                        gt=cell_gt
                    )

                    print(
                        f"[{run_name}] SpaGCN-binned-raw | "
                        f"Bin-ARI={bin_ari:.6f} | Bin-NMI={bin_nmi:.6f} | "
                        f"CellProj-ARI={cell_ari:.6f} | CellProj-NMI={cell_nmi:.6f}"
                    )

                    out_h5ad = os.path.join(OUT_DIR, f"{run_name}__SpaGCN_binned_raw.h5ad")
                    adata_bin.write_h5ad(out_h5ad)

                    results.append({
                        "Dataset": dataset_name,
                        "Section": section_name,
                        "Method": "SpaGCN_binned_raw",
                        "Seed": SEED,
                        "Label_Col": label_col,
                        "Slice_Col": slice_col,
                        "Spatial_Source": spatial_source,
                        "Bin_Size": BIN_SIZE,
                        "Min_Cells_Per_Bin": MIN_CELLS_PER_BIN,
                        "Recommended_l": float(l),
                        "Recommended_res": float(res),
                        "N_Clusters": int(n_clusters),
                        "N_Cells_Raw": int(adata_sec.n_obs),
                        "N_Bins": int(adata_bin.n_obs),
                        "N_Bin_Eval": int(n_bin_eval),
                        "N_Cell_Eval": int(n_cell_eval),
                        "Bin_ARI": float(bin_ari),
                        "Bin_NMI": float(bin_nmi),
                        "CellProj_ARI": float(cell_ari),
                        "CellProj_NMI": float(cell_nmi),
                        "Output_h5ad": out_h5ad,
                        "Status": "OK",
                        "Error": "",
                    })

                except Exception as e:
                    print(f"[{run_name}] FAILED: {e}")
                    results.append({
                        "Dataset": dataset_name,
                        "Section": section_name,
                        "Method": "SpaGCN_binned_raw",
                        "Seed": SEED,
                        "Label_Col": FORCE_LABEL_COL,
                        "Slice_Col": FORCE_SLICE_COL,
                        "Spatial_Source": "",
                        "Bin_Size": BIN_SIZE,
                        "Min_Cells_Per_Bin": MIN_CELLS_PER_BIN,
                        "Recommended_l": np.nan,
                        "Recommended_res": np.nan,
                        "N_Clusters": np.nan,
                        "N_Cells_Raw": np.nan,
                        "N_Bins": np.nan,
                        "N_Bin_Eval": np.nan,
                        "N_Cell_Eval": np.nan,
                        "Bin_ARI": np.nan,
                        "Bin_NMI": np.nan,
                        "CellProj_ARI": np.nan,
                        "CellProj_NMI": np.nan,
                        "Output_h5ad": "",
                        "Status": "FAILED",
                        "Error": str(e),
                    })

                pd.DataFrame(results).to_csv(RESULT_CSV, index=False)

        except Exception as e:
            print(f"[{dataset_name}] DATASET FAILED: {e}")
            results.append({
                "Dataset": dataset_name,
                "Section": "",
                "Method": "SpaGCN_binned_raw",
                "Seed": SEED,
                "Label_Col": FORCE_LABEL_COL,
                "Slice_Col": FORCE_SLICE_COL,
                "Spatial_Source": "",
                "Bin_Size": BIN_SIZE,
                "Min_Cells_Per_Bin": MIN_CELLS_PER_BIN,
                "Recommended_l": np.nan,
                "Recommended_res": np.nan,
                "N_Clusters": np.nan,
                "N_Cells_Raw": np.nan,
                "N_Bins": np.nan,
                "N_Bin_Eval": np.nan,
                "N_Cell_Eval": np.nan,
                "Bin_ARI": np.nan,
                "Bin_NMI": np.nan,
                "CellProj_ARI": np.nan,
                "CellProj_NMI": np.nan,
                "Output_h5ad": "",
                "Status": "FAILED",
                "Error": str(e),
            })
            pd.DataFrame(results).to_csv(RESULT_CSV, index=False)

    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULT_CSV, index=False)
    print("\nSaved:", RESULT_CSV)
    print(results_df)


if __name__ == "__main__":
    main()