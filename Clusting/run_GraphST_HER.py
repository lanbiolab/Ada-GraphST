import os
import sys
import glob
import random
import warnings
from collections import OrderedDict

warnings.filterwarnings("ignore")

# 如果你的 graphst 环境里有 R
os.environ["R_HOME"] = "/data2/liangyefeng/miniconda3/envs/graphst/lib/R"

PROJECT_ROOT = "/data2/liangyefeng/My_GraphST_Innovation"
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import torch
from scipy import sparse
from sklearn import metrics
from sklearn.decomposition import PCA

from GraphST.GraphST import GraphST
import GraphST.utils as graphst_utils


# =========================================================
# 0. 配置
# =========================================================
DATA_ROOT = "/data2/liangyefeng/My_GraphST_Innovation/data/Her2_tumor"
RESULT_ROOT = "/data2/liangyefeng/My_GraphST_Innovation/Result_HER2ST_GraphST_ParamSearch"
SEED = 41

# True: 自动只跑有 pathologist 标签的切片
ANNOTATED_ONLY = True

# True: 只跑 baseline 做测试
ONLY_RUN_BASELINE = False

# True: 跑完整 241 候选池
RUN_FULL_POOL = True

os.makedirs(RESULT_ROOT, exist_ok=True)


# =========================================================
# 1. 固定随机种子
# =========================================================
def reset_seed(seed=41):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


reset_seed(SEED)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device:", device)


# =========================================================
# 2. mclust
# =========================================================
def patched_mclust_R(adata, num_cluster, modelNames="EEE", used_obsm="emb_pca", random_seed=2020):
    import numpy as np
    import pandas as pd
    import rpy2.robjects as robjects
    from rpy2.robjects import r
    from rpy2.robjects import numpy2ri

    x = np.asarray(adata.obsm[used_obsm], dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"{used_obsm} must be 2D, but got shape {x.shape}")

    if not np.isfinite(x).all():
        bad_count = np.size(x) - np.isfinite(x).sum()
        raise ValueError(f"{used_obsm} contains NaN/Inf values, bad_count={bad_count}")

    print(f"[patched_mclust_R] used_obsm={used_obsm}, shape={x.shape}, dtype={x.dtype}")

    numpy2ri.activate()
    robjects.globalenv["x_mat"] = x
    robjects.globalenv["n_cluster"] = int(num_cluster)
    robjects.globalenv["model_name"] = modelNames
    robjects.globalenv["seed"] = int(random_seed)

    r(
        """
        suppressMessages(library(mclust))
        set.seed(seed)
        x_mat <- as.matrix(x_mat)
        dimnames(x_mat) <- NULL

        res <- tryCatch(
          Mclust(x_mat, G=n_cluster, modelNames=model_name),
          error = function(e) NULL
        )

        if (is.null(res)) {
          cls <- rep(NA, nrow(x_mat))
        } else {
          cls <- res$classification
          if (is.null(cls)) {
            cls <- rep(NA, nrow(x_mat))
          }
        }
        """
    )

    mclust_res = np.array(r["cls"], dtype=object)
    cls_series = pd.Series(mclust_res, index=adata.obs_names)
    cls_series = pd.to_numeric(cls_series, errors="coerce")

    na_count = cls_series.isna().sum()
    if na_count > 0:
        raise ValueError(f"mclust returned {na_count} invalid labels for {used_obsm}")

    adata.obs["mclust"] = cls_series.astype(int).astype("category")
    return adata


graphst_utils.mclust_R = patched_mclust_R


def run_mclust(adata, n_clusters, used_obsm="emb_pca", random_seed=41):
    adata = patched_mclust_R(
        adata,
        num_cluster=n_clusters,
        used_obsm=used_obsm,
        random_seed=random_seed
    )
    adata.obs["domain"] = adata.obs["mclust"].astype(str).astype("category")
    return adata


# =========================================================
# 3. HER2ST 读数函数
# =========================================================
def read_table_flex(path):
    """
    尝试 tab / csv 两种读法，优先有表头
    """
    for sep in ["\t", ","]:
        try:
            df = pd.read_csv(path, sep=sep)
            if df.shape[1] >= 2:
                return df
        except Exception:
            pass

    for sep in ["\t", ","]:
        try:
            df = pd.read_csv(path, sep=sep, header=None)
            if df.shape[1] >= 2:
                return df
        except Exception:
            pass

    raise ValueError(f"Failed to read table: {path}")


def find_spotfile(sample_id):
    patterns = [
        os.path.join(DATA_ROOT, "ST-spotfiles", f"{sample_id}_selection.tsv"),
        os.path.join(DATA_ROOT, "ST-spotfiles", f"{sample_id}_selection.tsv.gz"),
        os.path.join(DATA_ROOT, "ST-spotfiles", "**", f"{sample_id}_selection.tsv"),
        os.path.join(DATA_ROOT, "ST-spotfiles", "**", f"{sample_id}_selection.tsv.gz"),
    ]
    for p in patterns:
        matches = glob.glob(p, recursive=True)
        if len(matches) > 0:
            return sorted(matches)[0]
    return None


def find_patfile(sample_id):
    patterns = [
        os.path.join(DATA_ROOT, "ST-pat", f"{sample_id}_labeled_coordinates.tsv"),
        os.path.join(DATA_ROOT, "ST-pat", f"{sample_id}_labeled_coordinates.tsv.gz"),
        os.path.join(DATA_ROOT, "ST-pat", "lbl", f"{sample_id}_labeled_coordinates.tsv"),
        os.path.join(DATA_ROOT, "ST-pat", "lbl", f"{sample_id}_labeled_coordinates.tsv.gz"),
        os.path.join(DATA_ROOT, "ST-pat", "**", f"{sample_id}_labeled_coordinates.tsv"),
        os.path.join(DATA_ROOT, "ST-pat", "**", f"{sample_id}_labeled_coordinates.tsv.gz"),
    ]
    for p in patterns:
        matches = glob.glob(p, recursive=True)
        if len(matches) > 0:
            return sorted(matches)[0]
    return None


def discover_annotated_samples():
    # 自动扫描 ST-pat
    pat_matches = glob.glob(os.path.join(DATA_ROOT, "ST-pat", "**", "*_labeled_coordinates.tsv*"), recursive=True)
    if len(pat_matches) > 0:
        sample_ids = sorted(
            list(
                set(
                    os.path.basename(p).split("_labeled_coordinates")[0]
                    for p in pat_matches
                )
            )
        )
        return sample_ids

    # 兜底：常用 8 个 annotated sections
    return ["A1", "B1", "C1", "D1", "E1", "F1", "G2", "H1"]


def read_count_matrix(sample_id):
    path_plain = os.path.join(DATA_ROOT, "ST-cnts", f"{sample_id}.tsv")
    path_gz = os.path.join(DATA_ROOT, "ST-cnts", f"{sample_id}.tsv.gz")

    if os.path.exists(path_plain):
        cnt_path = path_plain
    elif os.path.exists(path_gz):
        cnt_path = path_gz
    else:
        raise FileNotFoundError(f"Count matrix not found for {sample_id}")

    cnt = pd.read_csv(cnt_path, sep="\t", index_col=0)

    # 去掉可能空列
    cnt = cnt.loc[:, cnt.columns.notnull()]
    cnt.index = cnt.index.astype(str)
    cnt.columns = cnt.columns.astype(str)

    return cnt, cnt_path


def standardize_spotfile_df(df, sample_id, n_obs_expected=None, obs_names=None):
    """
    从 ST-spotfiles 里尽量抽出:
    - x / y        阵列坐标（用于和 pat 标签对齐）
    - pixel_x/y    像素坐标（用于 GraphST spatial）
    """
    df = df.copy()

    # 如果第一列像 spot_id / index，就设为 index
    first_col = df.columns[0]
    if obs_names is not None:
        overlap_first = len(set(df[first_col].astype(str)) & set(map(str, obs_names)))
        if overlap_first > 0:
            df = df.set_index(first_col)

    # 优先重命名常见列
    rename_map = {}
    for c in df.columns:
        cl = str(c).lower()

        if cl in ["x", "adj_x", "array_row", "row", "array_x"]:
            rename_map[c] = "x"
        elif cl in ["y", "adj_y", "array_col", "col", "array_y"]:
            rename_map[c] = "y"
        elif cl in ["pixel_x", "xcoord", "x_coord", "imagecol", "pxl_col_in_fullres"]:
            rename_map[c] = "pixel_x"
        elif cl in ["pixel_y", "ycoord", "y_coord", "imagerow", "pxl_row_in_fullres"]:
            rename_map[c] = "pixel_y"

    df = df.rename(columns=rename_map)

    # 如果还没识别齐，就从数值列猜
    numeric_cols = []
    for c in df.columns:
        try:
            pd.to_numeric(df[c])
            numeric_cols.append(c)
        except Exception:
            pass

    if "x" not in df.columns or "y" not in df.columns:
        # 常见 spotfile 是 x,y,pixel_x,pixel_y
        if len(numeric_cols) >= 2:
            if "x" not in df.columns:
                df["x"] = pd.to_numeric(df[numeric_cols[0]], errors="coerce")
            if "y" not in df.columns:
                df["y"] = pd.to_numeric(df[numeric_cols[1]], errors="coerce")

    if "pixel_x" not in df.columns or "pixel_y" not in df.columns:
        if len(numeric_cols) >= 4:
            if "pixel_x" not in df.columns:
                df["pixel_x"] = pd.to_numeric(df[numeric_cols[2]], errors="coerce")
            if "pixel_y" not in df.columns:
                df["pixel_y"] = pd.to_numeric(df[numeric_cols[3]], errors="coerce")
        else:
            # 兜底：没有像素坐标就用阵列坐标
            if "pixel_x" not in df.columns and "x" in df.columns:
                df["pixel_x"] = pd.to_numeric(df["x"], errors="coerce")
            if "pixel_y" not in df.columns and "y" in df.columns:
                df["pixel_y"] = pd.to_numeric(df["y"], errors="coerce")

    # 统一类型
    for c in ["x", "y", "pixel_x", "pixel_y"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # 如果 index 还没对上，但是行数一致，则按顺序用
    if obs_names is not None:
        if not np.array_equal(df.index.astype(str).values, np.array(obs_names).astype(str)):
            if len(df) == len(obs_names):
                df.index = pd.Index(obs_names, name="spot_id")

    df["sample"] = sample_id
    return df


def read_pat_labels(sample_id):
    pat_path = find_patfile(sample_id)
    if pat_path is None:
        return None, None

    anno = pd.read_csv(pat_path, sep="\t")

    # 官方 SpatialPCA 页面里的列顺序
    # Row.names, adj_x, adj_y, pixel_x, pixel_y, label
    if anno.shape[1] >= 6:
        anno = anno.iloc[:, :6].copy()
        anno.columns = ["spot_id_raw", "adj_x", "adj_y", "pixel_x", "pixel_y", "label"]
    else:
        # 兜底：尽量猜
        cols = list(anno.columns)
        rename_map = {}
        for c in cols:
            cl = str(c).lower()
            if "label" in cl or "region" in cl or "annotation" in cl:
                rename_map[c] = "label"
            elif "adj_x" in cl or cl == "x":
                rename_map[c] = "adj_x"
            elif "adj_y" in cl or cl == "y":
                rename_map[c] = "adj_y"
            elif "pixel_x" in cl or "xcoord" in cl:
                rename_map[c] = "pixel_x"
            elif "pixel_y" in cl or "ycoord" in cl:
                rename_map[c] = "pixel_y"
        anno = anno.rename(columns=rename_map)

    anno["x"] = np.round(pd.to_numeric(anno["adj_x"], errors="coerce")).astype("Int64")
    anno["y"] = np.round(pd.to_numeric(anno["adj_y"], errors="coerce")).astype("Int64")
    anno["label"] = anno["label"].astype(str)

    return anno, pat_path


def build_adata_from_her2st(sample_id):
    cnt_df, cnt_path = read_count_matrix(sample_id)

    spot_path = find_spotfile(sample_id)
    if spot_path is None:
        raise FileNotFoundError(f"Spotfile not found for {sample_id}")

    spot_df = read_table_flex(spot_path)

    # 如果 spot 数跟 counts 列数相同但跟 counts 行数不同，自动转置
    if len(spot_df) == cnt_df.shape[1] and len(spot_df) != cnt_df.shape[0]:
        cnt_df = cnt_df.T
        cnt_df.index = cnt_df.index.astype(str)
        cnt_df.columns = cnt_df.columns.astype(str)

    spot_df = standardize_spotfile_df(
        spot_df,
        sample_id=sample_id,
        n_obs_expected=cnt_df.shape[0],
        obs_names=cnt_df.index.tolist()
    )

    # 若 spot_df 仍无法对齐，则按顺序兜底
    # 先尽量按 barcode/spot_id 对齐，而不是死卡行数
    cnt_index = cnt_df.index.astype(str)
    spot_df = spot_df.copy()
    spot_df.index = spot_df.index.astype(str)

    # 去重
    if spot_df.index.duplicated().sum() > 0:
        print(f"[{sample_id}] spot_df duplicated index:", int(spot_df.index.duplicated().sum()))
        spot_df = spot_df.loc[~spot_df.index.duplicated(keep="first")].copy()

    # 直接按 index 求交集
    common = cnt_index.intersection(spot_df.index)

    # 如果直接交集不够，再尝试从 spot_df 某一列里找 barcode
    if len(common) < min(len(cnt_index), len(spot_df)) * 0.8:
        best_col = None
        best_overlap = len(common)

        for c in spot_df.columns:
            vals = spot_df[c].astype(str)
            overlap = len(set(vals) & set(cnt_index))
            if overlap > best_overlap:
                best_overlap = overlap
                best_col = c

        if best_col is not None:
            print(f"[{sample_id}] use spot column [{best_col}] as barcode index, overlap={best_overlap}")
            tmp = spot_df.copy()
            tmp.index = tmp[best_col].astype(str)
            tmp = tmp.loc[~tmp.index.duplicated(keep='first')].copy()
            common = cnt_index.intersection(tmp.index)
            if len(common) > 0:
                spot_df = tmp

    print(f"[{sample_id}] count rows={cnt_df.shape[0]}, spot rows={spot_df.shape[0]}, common={len(common)}")

    # 如果大部分都能对上，就按共同 spot 对齐
    if len(common) >= max(10, min(cnt_df.shape[0], spot_df.shape[0]) - 5):
        cnt_df = cnt_df.loc[common].copy()
        spot_df = spot_df.loc[common].copy()
    else:
        # 最后兜底：如果只差很少，且行数接近，按顺序截断
        row_diff = abs(len(spot_df) - cnt_df.shape[0])
        if row_diff <= 5:
            print(f"[{sample_id}] fallback to truncation alignment, row_diff={row_diff}")
            n = min(len(spot_df), cnt_df.shape[0])
            cnt_df = cnt_df.iloc[:n, :].copy()
            spot_df = spot_df.iloc[:n, :].copy()
            spot_df.index = cnt_df.index.copy()
        else:
            raise ValueError(
                f"Cannot align count matrix and spotfile for {sample_id}: "
                f"count rows={cnt_df.shape[0]}, spot rows={len(spot_df)}, common={len(common)}"
            )

    adata = ad.AnnData(
        X=sparse.csr_matrix(cnt_df.values.astype(np.float32)),
        obs=spot_df.copy(),
        var=pd.DataFrame(index=cnt_df.columns.astype(str))
    )
    adata.obs_names = cnt_df.index.astype(str)
    adata.var_names = cnt_df.columns.astype(str)

    # spatial 用像素坐标
    adata.obsm["spatial"] = adata.obs[["pixel_x", "pixel_y"]].to_numpy(dtype=float)

    # 读 pathologist 标签
    anno_df, pat_path = read_pat_labels(sample_id)
    if anno_df is not None:
        adata.obs["x_round"] = np.round(pd.to_numeric(adata.obs["x"], errors="coerce")).astype("Int64")
        adata.obs["y_round"] = np.round(pd.to_numeric(adata.obs["y"], errors="coerce")).astype("Int64")

        obs_tmp = adata.obs.reset_index().rename(columns={"index": "spot_id"})
        merge_cols = ["x_round", "y_round"]
        anno_df = anno_df.rename(columns={"x": "x_round", "y": "y_round"})

        merged = obs_tmp.merge(
            anno_df[["x_round", "y_round", "label"]].drop_duplicates(),
            on=merge_cols,
            how="left"
        )

        merged = merged.set_index("spot_id")
        adata.obs["ground_truth"] = merged.loc[adata.obs_names, "label"].astype("category")
    else:
        pat_path = None
        adata.obs["ground_truth"] = pd.Series([pd.NA] * adata.n_obs, index=adata.obs_names, dtype="object")

    adata.uns["sample_id"] = sample_id
    adata.uns["cnt_path"] = cnt_path
    adata.uns["spot_path"] = spot_path
    adata.uns["pat_path"] = pat_path

    return adata


# =========================================================
# 4. 数据预处理
# =========================================================
def subset_hvg_keep_counts(adata, n_top_genes=3000, min_cells=3):
    adata = adata.copy()

    # 先过滤基因
    gene_mask = np.array((adata.X > 0).sum(axis=0)).ravel() >= min_cells
    if gene_mask.sum() == 0:
        raise ValueError("No genes pass min_cells filtering.")
    adata = adata[:, gene_mask].copy()

    tmp = adata.copy()
    sc.pp.normalize_total(tmp, target_sum=1e4)
    sc.pp.log1p(tmp)
    sc.pp.highly_variable_genes(
        tmp,
        flavor="seurat",
        n_top_genes=min(n_top_genes, tmp.n_vars)
    )

    if "highly_variable" not in tmp.var.columns or tmp.var["highly_variable"].sum() == 0:
        return adata

    hvg = tmp.var_names[tmp.var["highly_variable"]].tolist()
    adata = adata[:, adata.var_names.isin(hvg)].copy()

    # 恢复 counts 为 csr
    adata.X = sparse.csr_matrix(adata.X)
    return adata


# =========================================================
# 5. GraphST 候选池
# =========================================================
common_params = {
    "device": device,
    "random_seed": SEED,
    "epochs": 600,
    "dim_output": 64,
    "datatype": "Stereo",
    "warmup_epochs": 200,
    "update_interval": 20,
    "graph_update_rate": 0.3,
    "gamma": 3.0,
    "graph_reg_weight": 0.0,
    "use_learnable_proj": False,
}


def fmt_num(x):
    s = str(x)
    s = s.replace(".", "p")
    s = s.replace("-", "m")
    return s


candidate_dict = OrderedDict()
KEY_FIELDS = [
    "w_smooth",
    "w_sharpen",
    "gamma",
    "warmup_epochs",
    "update_interval",
    "graph_update_rate",
    "graph_reg_weight",
    "use_learnable_proj",
]


def add_candidate(name, use_original_baseline=False, **updates):
    cand = dict(common_params)
    cand.update(updates)

    if cand.get("w_sharpen", 0.0) == 0.0:
        cand["gamma"] = 3.0
        cand["warmup_epochs"] = 200
        cand["update_interval"] = 20
        cand["graph_update_rate"] = 0.3
        cand["graph_reg_weight"] = 0.0
        cand["use_learnable_proj"] = False

    key = tuple(cand.get(k) for k in KEY_FIELDS)
    if key not in candidate_dict:
        cand["name"] = name
        cand["use_original_baseline"] = use_original_baseline
        candidate_dict[key] = cand


# A. Baseline
add_candidate(
    "Baseline_Internal",
    use_original_baseline=True,
    w_smooth=0.0,
    w_sharpen=0.0,
    use_learnable_proj=False,
)

if RUN_FULL_POOL:
    # B. Smooth-only
    smooth_pool = [0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.30]
    for ws in smooth_pool:
        add_candidate(
            f"Smooth_{fmt_num(ws)}",
            w_smooth=ws,
            w_sharpen=0.0,
            use_learnable_proj=False,
        )

    # C. Sharpen-only
    sharpen_pool = [0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.50, 0.70]

    for hs in sharpen_pool:
        if hs <= 0.03:
            gamma_pool = [1.5, 2.0, 2.5]
        elif hs <= 0.10:
            gamma_pool = [1.5, 2.0, 2.5, 3.0]
        else:
            gamma_pool = [2.0, 2.5, 3.0, 4.0]

        for g in gamma_pool:
            add_candidate(
                f"Sharpen_{fmt_num(hs)}_g{fmt_num(g)}",
                w_smooth=0.0,
                w_sharpen=hs,
                gamma=g,
                use_learnable_proj=False,
            )

    for hs in [0.05, 0.08, 0.10]:
        for g in [2.5, 3.0]:
            add_candidate(
                f"Sharpen_{fmt_num(hs)}_g{fmt_num(g)}_Proj",
                w_smooth=0.0,
                w_sharpen=hs,
                gamma=g,
                use_learnable_proj=True,
            )

    # D. Hybrid
    hybrid_smooth_pool = [0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.12]
    hybrid_sharpen_pool = [0.01, 0.02, 0.03, 0.05, 0.08, 0.10]
    hybrid_gamma_pool = [2.0, 2.5, 3.0]

    for ws in hybrid_smooth_pool:
        for hs in hybrid_sharpen_pool:
            if ws >= 0.12 and hs >= 0.08:
                continue
            if ws >= 0.10 and hs >= 0.10:
                continue

            if hs <= 0.02:
                gamma_choices = [1.5, 2.0]
            else:
                gamma_choices = hybrid_gamma_pool

            for g in gamma_choices:
                add_candidate(
                    f"Hybrid_{fmt_num(ws)}_{fmt_num(hs)}_g{fmt_num(g)}",
                    w_smooth=ws,
                    w_sharpen=hs,
                    gamma=g,
                    use_learnable_proj=False,
                )

    # E. 次级扰动
    anchor_configs = [
        {"ws": 0.03, "hs": 0.05, "g": 2.0},
        {"ws": 0.05, "hs": 0.05, "g": 2.5},
        {"ws": 0.05, "hs": 0.08, "g": 2.5},
        {"ws": 0.08, "hs": 0.10, "g": 2.5},
    ]

    for a in anchor_configs:
        for wup in [100, 150, 250, 300]:
            add_candidate(
                f"Hybrid_{fmt_num(a['ws'])}_{fmt_num(a['hs'])}_g{fmt_num(a['g'])}_W{wup}",
                w_smooth=a["ws"],
                w_sharpen=a["hs"],
                gamma=a["g"],
                warmup_epochs=wup,
                use_learnable_proj=False,
            )

    for a in anchor_configs:
        for upd in [5, 10, 30, 50, 100]:
            add_candidate(
                f"Hybrid_{fmt_num(a['ws'])}_{fmt_num(a['hs'])}_g{fmt_num(a['g'])}_U{upd}",
                w_smooth=a["ws"],
                w_sharpen=a["hs"],
                gamma=a["g"],
                update_interval=upd,
                use_learnable_proj=False,
            )

    for a in anchor_configs:
        for ema in [0.05, 0.10, 0.20, 0.50, 0.70]:
            add_candidate(
                f"Hybrid_{fmt_num(a['ws'])}_{fmt_num(a['hs'])}_g{fmt_num(a['g'])}_E{fmt_num(ema)}",
                w_smooth=a["ws"],
                w_sharpen=a["hs"],
                gamma=a["g"],
                graph_update_rate=ema,
                use_learnable_proj=False,
            )

    for a in anchor_configs:
        for reg in [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]:
            add_candidate(
                f"Hybrid_{fmt_num(a['ws'])}_{fmt_num(a['hs'])}_g{fmt_num(a['g'])}_R{fmt_num(reg)}",
                w_smooth=a["ws"],
                w_sharpen=a["hs"],
                gamma=a["g"],
                graph_reg_weight=reg,
                use_learnable_proj=False,
            )

    for a in anchor_configs:
        add_candidate(
            f"Hybrid_{fmt_num(a['ws'])}_{fmt_num(a['hs'])}_g{fmt_num(a['g'])}_Proj",
            w_smooth=a["ws"],
            w_sharpen=a["hs"],
            gamma=a["g"],
            use_learnable_proj=True,
        )

candidates = list(candidate_dict.values())
if ONLY_RUN_BASELINE:
    candidates = [c for c in candidates if c["name"] == "Baseline_Internal"]

print(f"Total candidates: {len(candidates)}")

candidate_table = pd.DataFrame(candidates)
candidate_table.to_csv(
    os.path.join(RESULT_ROOT, "CandidatePool_Definition.csv"),
    index=False
)
print("Saved candidate pool definition.")


# =========================================================
# 6. 样本列表
# =========================================================
all_samples = sorted(
    [
        os.path.splitext(os.path.basename(p))[0].replace(".tsv", "")
        for p in glob.glob(os.path.join(DATA_ROOT, "ST-cnts", "*.tsv"))
    ]
)
all_samples = [s for s in all_samples if len(s) >= 2]

annotated_samples = discover_annotated_samples()
dataset_list = annotated_samples if ANNOTATED_ONLY else all_samples

print("Samples to run:", dataset_list)


# =========================================================
# 7. 跑一个候选
# =========================================================
def run_one_candidate(adata_input, candidate, n_clusters):
    reset_seed(SEED)
    adata = adata_input.copy()

    if candidate.get("use_original_baseline", False):
        print("[INFO] Using ORIGINAL GraphST baseline path")
        model = GraphST(adata, device=device)
    else:
        graphst_params = {
            k: v for k, v in candidate.items()
            if k not in ["name", "use_original_baseline"]
        }
        model = GraphST(adata, **graphst_params)

    adata = model.train()

    if "emb" not in adata.obsm:
        raise KeyError("adata.obsm['emb'] not found after training.")

    n_pcs = min(20, adata.obsm["emb"].shape[1])
    adata.obsm["emb_pca"] = PCA(
        n_components=n_pcs,
        random_state=SEED
    ).fit_transform(adata.obsm["emb"])

    adata = run_mclust(
        adata,
        n_clusters=n_clusters,
        used_obsm="emb_pca",
        random_seed=SEED
    )

    gt = adata.obs["ground_truth"]
    pred = adata.obs["domain"]

    mask = ~pd.isnull(gt)
    ari = metrics.adjusted_rand_score(gt[mask], pred[mask])
    nmi = metrics.normalized_mutual_info_score(gt[mask], pred[mask])

    return ari, nmi, int(mask.sum()), adata


# =========================================================
# 8. 主循环
# =========================================================
all_records = []
summary_records = []

for sample_id in dataset_list:
    print("\n" + "=" * 120)
    print(f"START SAMPLE: {sample_id}")
    print("=" * 120)

    dataset_save_dir = os.path.join(RESULT_ROOT, sample_id)
    os.makedirs(dataset_save_dir, exist_ok=True)

    try:
        adata_raw = build_adata_from_her2st(sample_id)
        if adata_raw.obs["ground_truth"].notna().sum() == 0:
            raise ValueError(f"No ground_truth labels found for {sample_id}")

        adata_raw = subset_hvg_keep_counts(adata_raw, n_top_genes=3000, min_cells=3)

        n_clusters = adata_raw.obs["ground_truth"].dropna().nunique()

        print(f"[Sample {sample_id}] shape after HVG: {adata_raw.shape}")
        print(f"[Sample {sample_id}] n_clusters from GT: {n_clusters}")
        print(f"[Sample {sample_id}] labeled spots: {adata_raw.obs['ground_truth'].notna().sum()}")
        print(f"[Sample {sample_id}] GT counts:")
        print(adata_raw.obs["ground_truth"].value_counts(dropna=False).head(20))
        print(f"[Sample {sample_id}] cnt_path: {adata_raw.uns['cnt_path']}")
        print(f"[Sample {sample_id}] spot_path: {adata_raw.uns['spot_path']}")
        print(f"[Sample {sample_id}] pat_path: {adata_raw.uns['pat_path']}")

    except Exception as e:
        print(f"Failed to prepare sample {sample_id}: {e}")

        for cand in candidates:
            record = dict(cand)
            record["sample_id"] = sample_id
            record["ARI"] = np.nan
            record["NMI"] = np.nan
            record["Error"] = f"prepare_failed: {e}"
            all_records.append(record)

        summary_records.append({
            "sample_id": sample_id,
            "best_name": None,
            "best_ARI": np.nan,
            "best_NMI": np.nan,
            "status": f"prepare_failed: {e}"
        })
        continue

    dataset_records = []
    best_ari = -1.0
    best_nmi = -1.0
    best_name = None

    for idx, candidate in enumerate(candidates, start=1):
        print(f"\n[{sample_id}] candidate {idx}/{len(candidates)} -> {candidate['name']}")

        try:
            ari, nmi, n_eval, adata_out = run_one_candidate(
                adata_input=adata_raw,
                candidate=candidate,
                n_clusters=n_clusters
            )

            record = dict(candidate)
            record["sample_id"] = sample_id
            record["N_Clusters"] = n_clusters
            record["N_Obs_Eval"] = n_eval
            record["ARI"] = ari
            record["NMI"] = nmi
            dataset_records.append(record)
            all_records.append(record)

            if (ari > best_ari) or (np.isclose(ari, best_ari) and nmi > best_nmi):
                best_ari = ari
                best_nmi = nmi
                best_name = candidate["name"]

                # 保存当前最佳 embedding / domain
                adata_out.write_h5ad(os.path.join(dataset_save_dir, f"{sample_id}_current_best.h5ad"))

            print(f"ARI={ari:.6f} | NMI={nmi:.6f}")

        except Exception as e:
            print(f"Candidate [{candidate['name']}] failed on {sample_id}: {e}")
            record = dict(candidate)
            record["sample_id"] = sample_id
            record["N_Clusters"] = n_clusters
            record["ARI"] = np.nan
            record["NMI"] = np.nan
            record["Error"] = str(e)
            dataset_records.append(record)
            all_records.append(record)
            continue

    dataset_df = pd.DataFrame(dataset_records)
    dataset_df["ARI_rank"] = dataset_df["ARI"].rank(ascending=False, method="min")
    dataset_df["NMI_rank"] = dataset_df["NMI"].rank(ascending=False, method="min")
    dataset_df = dataset_df.sort_values(by=["ARI", "NMI"], ascending=[False, False])

    dataset_csv = os.path.join(dataset_save_dir, f"{sample_id}_all_candidate_results.csv")
    dataset_df.to_csv(dataset_csv, index=False)

    top10_csv = os.path.join(dataset_save_dir, f"{sample_id}_top10_candidates.csv")
    dataset_df.head(10).to_csv(top10_csv, index=False)

    summary_records.append({
        "sample_id": sample_id,
        "best_name": best_name,
        "best_ARI": best_ari if best_ari >= 0 else np.nan,
        "best_NMI": best_nmi if best_nmi >= 0 else np.nan,
        "status": "success"
    })

    print("\n" + "-" * 100)
    print(f"BEST RESULT FOR {sample_id}")
    print(f"Best candidate: {best_name}")
    print(f"Best ARI: {best_ari:.6f}")
    print(f"Best NMI: {best_nmi:.6f}")
    print("-" * 100)


# =========================================================
# 9. 保存全局 CSV
# =========================================================
all_results_df = pd.DataFrame(all_records)
all_results_csv = os.path.join(RESULT_ROOT, "HER2ST_AllCandidate_LongResults.csv")
all_results_df.to_csv(all_results_csv, index=False)

ari_wide = all_results_df.pivot(index="sample_id", columns="name", values="ARI").reset_index()
ari_wide_csv = os.path.join(RESULT_ROOT, "HER2ST_ARI_Wide.csv")
ari_wide.to_csv(ari_wide_csv, index=False)

nmi_wide = all_results_df.pivot(index="sample_id", columns="name", values="NMI").reset_index()
nmi_wide_csv = os.path.join(RESULT_ROOT, "HER2ST_NMI_Wide.csv")
nmi_wide.to_csv(nmi_wide_csv, index=False)

candidate_summary_records = []
for cand_name, subdf in all_results_df.groupby("name"):
    candidate_summary_records.append({
        "name": cand_name,
        "mean_ARI": subdf["ARI"].mean(skipna=True),
        "median_ARI": subdf["ARI"].median(skipna=True),
        "std_ARI": subdf["ARI"].std(skipna=True),
        "mean_NMI": subdf["NMI"].mean(skipna=True),
        "median_NMI": subdf["NMI"].median(skipna=True),
        "std_NMI": subdf["NMI"].std(skipna=True),
        "num_valid_samples": int(subdf["ARI"].notna().sum()),
    })

candidate_summary_df = pd.DataFrame(candidate_summary_records).sort_values(
    by=["median_ARI", "mean_ARI", "mean_NMI"],
    ascending=[False, False, False]
)
candidate_summary_csv = os.path.join(RESULT_ROOT, "HER2ST_Candidate_Summary.csv")
candidate_summary_df.to_csv(candidate_summary_csv, index=False)

best_summary_df = pd.DataFrame(summary_records)
best_summary_csv = os.path.join(RESULT_ROOT, "HER2ST_Best_Summary.csv")
best_summary_df.to_csv(best_summary_csv, index=False)

print("\n" + "=" * 120)
print("HER2ST FINISHED")
print("=" * 120)
print(best_summary_df)

print(f"\nSaved candidate definition to: {os.path.join(RESULT_ROOT, 'CandidatePool_Definition.csv')}")
print(f"Saved long results to: {all_results_csv}")
print(f"Saved ARI wide table to: {ari_wide_csv}")
print(f"Saved NMI wide table to: {nmi_wide_csv}")
print(f"Saved candidate summary to: {candidate_summary_csv}")
print(f"Saved best summary to: {best_summary_csv}")