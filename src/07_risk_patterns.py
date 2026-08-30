from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# SQUAT DATASET — RISK-RELATED MOVEMENT INDICATORS
# FINAL PUBLICATION VERSION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

FEATURE_ROOT = ROOT / "data" / "processed" / "features"
RISK_ROOT = ROOT / "data" / "processed" / "risk"
GRAPH_ROOT = ROOT / "outputs" / "risk_graphs"

RISK_ROOT.mkdir(parents=True, exist_ok=True)
GRAPH_ROOT.mkdir(parents=True, exist_ok=True)


# ============================================================
# CLEAN OLD RISK OUTPUTS
# ============================================================

print("=" * 70)
print("CLEANING OLD RISK OUTPUTS")
print("=" * 70)

for file in GRAPH_ROOT.glob("*"):
    if file.is_file():
        file.unlink()

print("[DONE] Old risk graphs removed")


# ============================================================
# LOAD FEATURES
# ============================================================

def load_features(view):

    path = FEATURE_ROOT / f"{view}_features.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Feature file not found:\n{path}"
        )

    df = pd.read_csv(path)

    print()
    print("=" * 70)
    print(f"LOADING {view.upper()} FEATURES")
    print("=" * 70)

    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    return df


# ============================================================
# FORCE NUMERIC
# ============================================================

def numeric(df, column):

    return pd.to_numeric(
        df[column],
        errors="coerce"
    )


# ============================================================
# FRONT VIEW INDICATORS
# ============================================================

def calculate_front_indicators(df):

    result = df.copy()

    # --------------------------------------------------------
    # LEFT-RIGHT KNEE ASYMMETRY
    # --------------------------------------------------------

    result["knee_asymmetry_deg"] = (
        numeric(
            result,
            "left_knee_mean_deg"
        )
        -
        numeric(
            result,
            "right_knee_mean_deg"
        )
    ).abs()

    # --------------------------------------------------------
    # LEFT-RIGHT HIP ASYMMETRY
    # --------------------------------------------------------

    result["hip_asymmetry_deg"] = (
        numeric(
            result,
            "left_hip_mean_deg"
        )
        -
        numeric(
            result,
            "right_hip_mean_deg"
        )
    ).abs()

    # --------------------------------------------------------
    # LEFT-RIGHT ANKLE ASYMMETRY
    # --------------------------------------------------------

    result["ankle_asymmetry_deg"] = (
        numeric(
            result,
            "left_ankle_mean_deg"
        )
        -
        numeric(
            result,
            "right_ankle_mean_deg"
        )
    ).abs()

    return result


# ============================================================
# SIDE VIEW INDICATOR
# ============================================================

def calculate_side_indicators(df):

    result = df.copy()

    # Side view is treated as sagittal-plane movement.
    # Therefore we retain trunk excursion rather than
    # interpreting left/right asymmetry from this view.

    result["trunk_excursion_deg"] = numeric(
        result,
        "trunk_range_deg"
    )

    return result


# ============================================================
# CALCULATE DATASET-DERIVED 95TH PERCENTILE
# ============================================================

def percentile_95(series):

    values = pd.to_numeric(
        series,
        errors="coerce"
    ).dropna()

    if len(values) == 0:
        return np.nan

    return float(
        np.percentile(
            values,
            95
        )
    )


# ============================================================
# ADD UPPER 5% FLAGS
# ============================================================

def add_flag(
    df,
    column
):

    threshold = percentile_95(
        df[column]
    )

    flag_column = (
        column.replace(
            "_deg",
            ""
        )
        + "_upper5pct_flag"
    )

    if np.isnan(threshold):

        df[flag_column] = 0

    else:

        df[flag_column] = (
            pd.to_numeric(
                df[column],
                errors="coerce"
            )
            >= threshold
        ).astype(int)

    return df, threshold


# ============================================================
# FRONT ANALYSIS
# ============================================================

def process_front(df):

    print()
    print("=" * 70)
    print("FRONT VIEW — LEFT/RIGHT MOVEMENT INDICATORS")
    print("=" * 70)

    df = calculate_front_indicators(df)

    thresholds = {}

    for column in [
        "knee_asymmetry_deg",
        "hip_asymmetry_deg",
        "ankle_asymmetry_deg"
    ]:

        df, threshold = add_flag(
            df,
            column
        )

        thresholds[column] = threshold

        if np.isnan(threshold):

            print(
                f"{column}: no valid data"
            )

        else:

            print(
                f"{column}: "
                f"95th percentile = "
                f"{threshold:.2f}°"
            )

    return df, thresholds


# ============================================================
# SIDE ANALYSIS
# ============================================================

def process_side(df):

    print()
    print("=" * 70)
    print("SIDE VIEW — SAGITTAL MOVEMENT INDICATOR")
    print("=" * 70)

    df = calculate_side_indicators(df)

    df, threshold = add_flag(
        df,
        "trunk_excursion_deg"
    )

    if np.isnan(threshold):

        print(
            "trunk_excursion_deg: no valid data"
        )

    else:

        print(
            "trunk_excursion_deg: "
            f"95th percentile = "
            f"{threshold:.2f}°"
        )

    return df, {
        "trunk_excursion_deg": threshold
    }


# ============================================================
# SAVE FRONT TABLE
# ============================================================

def save_front_table(df):

    columns = [
        "unique_rep",
        "participant",
        "rep_id",

        "knee_asymmetry_deg",
        "hip_asymmetry_deg",
        "ankle_asymmetry_deg",

        "knee_asymmetry_upper5pct_flag",
        "hip_asymmetry_upper5pct_flag",
        "ankle_asymmetry_upper5pct_flag"
    ]

    columns = [
        c
        for c in columns
        if c in df.columns
    ]

    output = df[columns].copy()

    path = (
        RISK_ROOT
        / "front_risk_indicators.csv"
    )

    output.to_csv(
        path,
        index=False
    )

    print()
    print(
        f"[DONE] Saved: {path}"
    )

    print(
        f"Rows: {len(output)}"
    )

    print(
        f"Columns: {len(output.columns)}"
    )


# ============================================================
# SAVE SIDE TABLE
# ============================================================

def save_side_table(df):

    columns = [
        "unique_rep",
        "participant",
        "rep_id",

        "trunk_excursion_deg",
        "trunk_excursion_upper5pct_flag"
    ]

    columns = [
        c
        for c in columns
        if c in df.columns
    ]

    output = df[columns].copy()

    path = (
        RISK_ROOT
        / "side_risk_indicators.csv"
    )

    output.to_csv(
        path,
        index=False
    )

    print()
    print(
        f"[DONE] Saved: {path}"
    )

    print(
        f"Rows: {len(output)}"
    )

    print(
        f"Columns: {len(output.columns)}"
    )


# ============================================================
# DISTRIBUTION PLOT
# ============================================================

def create_distribution(
    df,
    column,
    title,
    xlabel,
    filename
):

    values = (
        pd.to_numeric(
            df[column],
            errors="coerce"
        )
        .dropna()
        .values
    )

    if len(values) == 0:

        print(
            f"[WARNING] No valid data: {column}"
        )

        return

    mean_value = float(
        np.mean(values)
    )

    median_value = float(
        np.median(values)
    )

    percentile_value = float(
        np.percentile(
            values,
            95
        )
    )

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.hist(
        values,
        bins=15,
        alpha=0.75,
        edgecolor="black"
    )

    ax.axvline(
        mean_value,
        linewidth=2,
        label=f"Mean = {mean_value:.2f}°"
    )

    ax.axvline(
        median_value,
        linestyle=":",
        linewidth=2,
        label=f"Median = {median_value:.2f}°"
    )

    ax.axvline(
        percentile_value,
        linestyle="--",
        linewidth=2,
        label=(
            f"95th percentile = "
            f"{percentile_value:.2f}°"
        )
    )

    ax.set_title(
        title,
        fontsize=16,
        fontweight="bold"
    )

    ax.set_xlabel(
        xlabel,
        fontsize=12
    )

    ax.set_ylabel(
        "Number of repetitions",
        fontsize=12
    )

    ax.grid(
        axis="y",
        alpha=0.20
    )

    ax.legend(
        frameon=False
    )

    fig.tight_layout()

    png_path = (
        GRAPH_ROOT
        / f"{filename}.png"
    )

    pdf_path = (
        GRAPH_ROOT
        / f"{filename}.pdf"
    )

    fig.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight"
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(
        f"[DONE] {png_path.name}"
    )

    print(
        f"[DONE] {pdf_path.name}"
    )


# ============================================================
# FRONT ASYMMETRY COMPARISON
# ============================================================

def create_front_asymmetry_comparison(df):

    columns = [
        "knee_asymmetry_deg",
        "hip_asymmetry_deg",
        "ankle_asymmetry_deg"
    ]

    labels = [
        "Knee",
        "Hip",
        "Ankle"
    ]

    means = []
    p95 = []

    for column in columns:

        values = (
            pd.to_numeric(
                df[column],
                errors="coerce"
            )
            .dropna()
        )

        means.append(
            values.mean()
        )

        p95.append(
            np.percentile(
                values,
                95
            )
        )

    x = np.arange(
        len(labels)
    )

    width = 0.35

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    ax.bar(
        x - width / 2,
        means,
        width,
        label="Mean"
    )

    ax.bar(
        x + width / 2,
        p95,
        width,
        label="95th percentile"
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        labels
    )

    ax.set_ylabel(
        "Absolute left-right angle difference (degrees)"
    )

    ax.set_title(
        "Front View — Joint Asymmetry Summary",
        fontsize=16,
        fontweight="bold"
    )

    ax.grid(
        axis="y",
        alpha=0.20
    )

    ax.legend(
        frameon=False
    )

    fig.tight_layout()

    png_path = (
        GRAPH_ROOT
        / "front_asymmetry_summary.png"
    )

    pdf_path = (
        GRAPH_ROOT
        / "front_asymmetry_summary.pdf"
    )

    fig.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight"
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(
        f"[DONE] {png_path.name}"
    )

    print(
        f"[DONE] {pdf_path.name}"
    )


# ============================================================
# SIDE TRUNK SUMMARY
# ============================================================

def create_side_trunk_summary(df):

    values = (
        pd.to_numeric(
            df["trunk_excursion_deg"],
            errors="coerce"
        )
        .dropna()
    )

    mean_value = values.mean()

    percentile_value = np.percentile(
        values,
        95
    )

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    ax.hist(
        values,
        bins=15,
        alpha=0.75,
        edgecolor="black"
    )

    ax.axvline(
        mean_value,
        linewidth=2,
        label=f"Mean = {mean_value:.2f}°"
    )

    ax.axvline(
        percentile_value,
        linestyle="--",
        linewidth=2,
        label=(
            f"95th percentile = "
            f"{percentile_value:.2f}°"
        )
    )

    ax.set_title(
        "Side View — Trunk Angular Excursion",
        fontsize=16,
        fontweight="bold"
    )

    ax.set_xlabel(
        "Trunk angular excursion (degrees)"
    )

    ax.set_ylabel(
        "Number of repetitions"
    )

    ax.grid(
        axis="y",
        alpha=0.20
    )

    ax.legend(
        frameon=False
    )

    fig.tight_layout()

    png_path = (
        GRAPH_ROOT
        / "side_trunk_excursion_distribution.png"
    )

    pdf_path = (
        GRAPH_ROOT
        / "side_trunk_excursion_distribution.pdf"
    )

    fig.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight"
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(
        f"[DONE] {png_path.name}"
    )

    print(
        f"[DONE] {pdf_path.name}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print(
        "SQUAT DATASET — FINAL MOVEMENT INDICATOR ANALYSIS"
    )
    print("=" * 70)

    # ========================================================
    # FRONT
    # ========================================================

    front = load_features(
        "front"
    )

    front, front_thresholds = process_front(
        front
    )

    save_front_table(
        front
    )

    # ========================================================
    # SIDE
    # ========================================================

    side = load_features(
        "side"
    )

    side, side_thresholds = process_side(
        side
    )

    save_side_table(
        side
    )

    # ========================================================
    # FRONT GRAPHS
    # ========================================================

    print()
    print("=" * 70)
    print(
        "GENERATING FRONT-VIEW INDICATOR GRAPHS"
    )
    print("=" * 70)

    create_distribution(
        front,
        "knee_asymmetry_deg",
        "Front View — Knee Left-Right Asymmetry",
        "Absolute left-right knee angle difference (degrees)",
        "front_knee_asymmetry_distribution"
    )

    create_distribution(
        front,
        "hip_asymmetry_deg",
        "Front View — Hip Left-Right Asymmetry",
        "Absolute left-right hip angle difference (degrees)",
        "front_hip_asymmetry_distribution"
    )

    create_distribution(
        front,
        "ankle_asymmetry_deg",
        "Front View — Ankle Left-Right Asymmetry",
        "Absolute left-right ankle angle difference (degrees)",
        "front_ankle_asymmetry_distribution"
    )

    create_front_asymmetry_comparison(
        front
    )

    # ========================================================
    # SIDE GRAPH
    # ========================================================

    print()
    print("=" * 70)
    print(
        "GENERATING SIDE-VIEW INDICATOR GRAPH"
    )
    print("=" * 70)

    create_side_trunk_summary(
        side
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print(
        "FINAL MOVEMENT INDICATOR GENERATION COMPLETE"
    )
    print("=" * 70)

    print()
    print("Risk / movement tables:")

    print(
        RISK_ROOT
    )

    print()
    print("Risk / movement graphs:")

    print(
        GRAPH_ROOT
    )

    print()
    print("FRONT VIEW:")

    for column, threshold in front_thresholds.items():

        if not np.isnan(threshold):

            count = int(
                (
                    front[column]
                    >= threshold
                ).sum()
            )

            print(
                f"{column}: "
                f"95th percentile = "
                f"{threshold:.2f}°, "
                f"upper 5% = "
                f"{count}/100"
            )

    print()
    print("SIDE VIEW:")

    threshold = side_thresholds[
        "trunk_excursion_deg"
    ]

    if not np.isnan(threshold):

        count = int(
            (
                side[
                    "trunk_excursion_deg"
                ]
                >= threshold
            ).sum()
        )

        print(
            "trunk_excursion_deg: "
            f"95th percentile = "
            f"{threshold:.2f}°, "
            f"upper 5% = "
            f"{count}/100"
        )

    print()
    print("=" * 70)
    print("SCIENTIFIC NOTE")
    print("=" * 70)

    print(
        "Front-view indicators describe projected "
        "left-right movement asymmetry."
    )

    print(
        "Side-view trunk excursion describes "
        "sagittal-plane movement."
    )

    print(
        "Upper-5% values are dataset-derived "
        "statistical flags."
    )

    print(
        "They are NOT clinical injury thresholds "
        "or injury diagnoses."
    )

    print()


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()