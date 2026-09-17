# ================================
# Combined Figure: Panel A (waveforms + SHAP example bars) + Panel B (confusion matrix)
# Same row, panel A left / panel B right.
# Reuses the exact same data/logic as plots_sep_17.py (waveforms) and
# model_shap_xlsx_no_metadata_permutation_no_scale_8_30_sep11.py (SHAP CSV + confusion matrix).
# ================================

import os
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from collections import defaultdict
from sklearn.preprocessing import MinMaxScaler

# ----------------
# CONFIG (identical to plots_sep_17.py)
# ----------------
BASE_PATH  = "/sessions/loving-lucid-newton/mnt/Moles/Data"
CLICKS_DIR = os.path.join(BASE_PATH, "clicks")

USED_MIN_COL = 8
USED_MAX_COL = 30
PEAK_COL = 21
SAMPLE_PERIOD_MS = 1.0
MAX_FILES = None

SHAP_CSV_PATH = "/sessions/loving-lucid-newton/mnt/outputs/rerun/plots/cols8_30_shap_weights.csv"
CONFMAT_PNG_PATH = ""

OUT_PNG = "/sessions/loving-lucid-newton/mnt/outputs/rerun/Figure4_waveform_shap_confmat_R1R5.png"
OUT_PDF = "/sessions/loving-lucid-newton/mnt/outputs/rerun/Figure4_waveform_shap_confmat_R1R5.pdf"

N_EXAMPLE_BARS = 5
BAR_MAX_HEIGHT = 0.20   # bars occupy the bottom 0-0.20 band of the 0-1 amplitude axis

plt.rcParams.update({
    "savefig.dpi": 300,
    "figure.dpi": 150,
    "axes.titlesize": 16,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "-",
    "font.family": "DejaVu Sans",
})

# ----------------
# HELPERS (same as plots_sep_17.py)
# ----------------
def parse_individual_from_filename(fname: str) -> str:
    base = os.path.basename(fname)
    m = re.match(r"^(R\d+)S\d+", base, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()
    m2 = re.match(r"^(R\d+)", base, flags=re.IGNORECASE)
    if m2:
        return m2.group(1).upper()
    return base.split('.')[0][:2].upper()

def load_xlsx_matrix(path: str) -> np.ndarray:
    df = pd.read_excel(path, header=None, engine="openpyxl")
    df = df.apply(pd.to_numeric, errors='coerce').dropna(how='all')
    df = df.dropna(axis=1, how='all')
    return df.to_numpy(dtype=float)

def rowwise_max_normalize(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    row_max = np.max(np.abs(X), axis=1, keepdims=True)
    row_max = np.where(row_max < eps, 1.0, row_max)
    return X / row_max

def cols_to_time_ms(cols_1based: np.ndarray) -> np.ndarray:
    return (cols_1based - PEAK_COL) * SAMPLE_PERIOD_MS

# ----------------
# Build per-mole averaged waveform curves (identical pipeline to plots_sep_17.py main())
# ----------------
def compute_mole_curves():
    files = [os.path.join(CLICKS_DIR, f) for f in os.listdir(CLICKS_DIR)
             if f.lower().endswith(".xlsx") and not os.path.basename(f).upper().startswith("R6")]
    files.sort()
    if MAX_FILES is not None:
        files = files[:MAX_FILES]
    if not files:
        raise RuntimeError(f"No .xlsx files found in {CLICKS_DIR}")

    max_cols_observed = 0
    window_slices_all = []
    mole_to_window_curves = defaultdict(list)
    mole_to_full_curves = defaultdict(list)

    for path in files:
        X = load_xlsx_matrix(path)
        if X.size == 0 or X.shape[1] <= USED_MAX_COL:
            continue
        max_cols_observed = max(max_cols_observed, X.shape[1])
        Xn = rowwise_max_normalize(X)
        Xw = Xn[:, USED_MIN_COL:USED_MAX_COL+1]
        window_slices_all.append(Xw)

    Xw_all = np.vstack(window_slices_all)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(Xw_all)

    for path in files:
        X = load_xlsx_matrix(path)
        if X.size == 0 or X.shape[1] <= USED_MAX_COL:
            continue
        Xn = rowwise_max_normalize(X)
        Xw = Xn[:, USED_MIN_COL:USED_MAX_COL+1]
        Xw_scaled = scaler.transform(Xw)
        mean_curve_window = np.nanmean(Xw_scaled, axis=0)
        mean_curve_full = np.nanmean(Xn, axis=0)
        indiv = parse_individual_from_filename(path)
        mole_to_window_curves[indiv].append(mean_curve_window)
        mole_to_full_curves[indiv].append(mean_curve_full)

    mole_full_avg = {}
    for mole, curves in mole_to_full_curves.items():
        F = np.vstack(curves)
        full_avg = F.mean(axis=0)
        w = full_avg[USED_MIN_COL:USED_MAX_COL+1]
        w_min, w_max = np.nanmin(w), np.nanmax(w)
        denom = (w_max - w_min) if (w_max - w_min) != 0 else 1.0
        full_rescaled = (full_avg - w_min) / denom
        mole_full_avg[mole] = full_rescaled

    return mole_full_avg, max_cols_observed

# ----------------
# Load SHAP importance for ALL columns (8-30), normalized
# ----------------
def load_top_shap_bars(n=N_EXAMPLE_BARS):
    df = pd.read_csv(SHAP_CSV_PATH)
    df = df.dropna(subset=["global_mean_abs_shap"]).copy()
    df["col_index"] = df["feature"].str.extract(r"(\d+)$").astype(int)
    df = df.sort_values("col_index").copy()
    df["time_ms"] = cols_to_time_ms(df["col_index"].to_numpy())
    max_shap = df["global_mean_abs_shap"].max()
    df["norm_height"] = (df["global_mean_abs_shap"] / max_shap) * BAR_MAX_HEIGHT
    return df

# ----------------
# MAIN
# ----------------
def main():
    mole_full_avg, max_cols_observed = compute_mole_curves()
    cols_1based_full = np.arange(1, max_cols_observed + 1)
    time_ms_full = cols_to_time_ms(cols_1based_full)
    top_shap = load_top_shap_bars()

    fig = plt.figure(figsize=(16.5, 6.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.38)
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])

    # ---------- Panel A: waveforms + SHAP example bars ----------
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key().get('color', None)
    for i, (mole, vec) in enumerate(sorted(mole_full_avg.items())):
        color = None if not color_cycle else color_cycle[i % len(color_cycle)]
        if vec.shape[0] < max_cols_observed:
            pad = np.full(max_cols_observed - vec.shape[0], np.nan)
            y = np.concatenate([vec, pad])
        else:
            y = vec[:max_cols_observed]
        axA.plot(time_ms_full, y, label=mole, linewidth=2.0, color=color, zorder=3)

    axA.axvline(0.0, color="k", linewidth=1.5, alpha=0.6, linestyle="--", zorder=2)
    win_left_ms  = cols_to_time_ms(np.array([USED_MIN_COL]))[0]
    win_right_ms = cols_to_time_ms(np.array([USED_MAX_COL]))[0]
    axA.axvspan(win_left_ms, win_right_ms, color="grey", alpha=0.08,
                label="Analysis window", zorder=1)

    # SHAP bars (all columns 8-30, normalized): sitting on the baseline (y=0), below the waveforms
    bar_width = 0.9
    for _, row in top_shap.iterrows():
        axA.add_patch(Rectangle(
            (row["time_ms"] - bar_width / 2, 0.0), bar_width, row["norm_height"],
            facecolor="#3b78c2", edgecolor="#1f4e8c", alpha=0.55, linewidth=0.6, zorder=2.5
        ))
    # one proxy handle for the legend
    axA.add_patch(Rectangle((0, 0), 0, 0, facecolor="#3b78c2", edgecolor="#1f4e8c",
                             alpha=0.55, label="SHAP importance (normalized)"))

    axA.set_title("Normalized Waveforms by Individual")
    axA.set_xlabel("Time relative to peak (ms)")
    axA.set_ylabel("Scaled amplitude (0–1)")
    axA.set_xlim(time_ms_full[0], time_ms_full[-1])
    axA.set_ylim(bottom=0)
    axA.grid(True, alpha=0.25)

    leg = axA.legend(
        title="Individual",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=4,
        frameon=False
    )
    for line in leg.get_lines():
        line.set_linewidth(3)

    span = time_ms_full[-1] - time_ms_full[0]
    step = max(1, int(round(span / 15.0)))
    ticks = np.arange(time_ms_full[0], time_ms_full[-1] + 1e-9, step)
    axA.set_xticks(ticks)

    axA.text(-0.06, 1.04, "A", transform=axA.transAxes, fontsize=20, fontweight="bold", va="top")

    # ---------- Panel B: confusion matrix ----------
    # Redrawn natively (not embedded as a raster) using the exact counts from
    # cols8_30_confusion_matrix.png, so that the title and the chart frame are
    # placed by the same matplotlib mechanism as Panel A -- this guarantees the
    # "A"/"B" labels, the two titles, and the two plot boxes all line up to the
    # same vertical level, since axA and axB share one GridSpec row.
    _cmdf = pd.read_csv("/sessions/loving-lucid-newton/mnt/outputs/rerun/plots/cols8_30_confusion_counts.csv", index_col=0)
    cm_labels = list(_cmdf.columns)
    cm = _cmdf.to_numpy()

    # Row-normalize to percentages (each row = true label sums to 100%)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_pct = 100.0 * cm / row_sums

    im = axB.imshow(cm_pct, cmap="Blues", aspect="auto", vmin=0, vmax=100)
    axB.grid(False)
    thresh = cm_pct.max() / 2.0
    for i in range(cm_pct.shape[0]):
        for j in range(cm_pct.shape[1]):
            val = cm_pct[i, j]
            txt_color = "white" if val > thresh else "black"
            axB.text(j, i, f"{val:.1f}%", ha="center", va="center",
                      color=txt_color, fontsize=10)

    axB.set_xticks(range(len(cm_labels)))
    axB.set_xticklabels(cm_labels)
    axB.set_yticks(range(len(cm_labels)))
    axB.set_yticklabels(cm_labels)
    axB.set_xlabel("Predicted label")
    axB.set_ylabel("True label")
    axB.set_title("Confusion Matrix (% of true label)")

    cbar = fig.colorbar(im, ax=axB, fraction=0.046, pad=0.04)
    cbar.set_label("% of true label")

    axB.text(-0.06, 1.04, "B", transform=axB.transAxes, fontsize=20, fontweight="bold", va="top")

    plt.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    plt.savefig(OUT_PDF, bbox_inches="tight")
    print(f"Saved combined figure to: {OUT_PNG}")
    print(f"Saved combined figure to: {OUT_PDF}")

if __name__ == "__main__":
    main()
