from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

FEATURE_ROOT = ROOT / "data" / "processed" / "features"
RISK_ROOT = ROOT / "data" / "processed" / "risk"
GRAPH_ROOT = ROOT / "outputs" / "risk_graphs"

RISK_ROOT.mkdir(parents=True, exist_ok=True)
GRAPH_ROOT.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD FEATURE DATA
# ============================================================

def load_features(view):

    path = FEATURE_ROOT / f"{view}_features.csv"

    print()
    print("=" * 70)
    print(f"LOADING {view.upper()} FEATURES")
    print("=" * 70)

    if not path.exists():
        raise FileNotFoundError(
            f"Feature file not found:\n{path}"
        )

    df = pd.read_csv(path)

    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    return df


# ============================================================
# NUMERIC HELPER
# ============================================================

def num(df, column):

    return pd.to_numeric(
        df[column],
        errors="coerce"
    )


# ============================================================
# CALCULATE RISK-RELATED MOVEMENT INDICATORS
# ============================================================

def calculate_indicators(df):

    result = df.copy()

    # --------------------------------------------------------
    # KNEE LEFT-RIGHT ASYMMETRY
    # --------------------------------------------------------

    result["knee_asymmetry_deg"] = (
        num(
            result,
            "left_knee_mean_deg"
        )
        -
        num(
            result,
            "right_knee_mean_deg"
        )
    ).abs()

    # --------------------------------------------------------
    # HIP LEFT-RIGHT ASYMMETRY
    # --------------------------------------------------------

    result["hip_asymmetry_deg"] = (
        num(
            result,
            "left_hip_mean_deg"
        )
        -
        num(
            result,
            "right_hip_mean_deg"
        )
    ).abs()

    # --------------------------------------------------------
    # ANKLE LEFT-RIGHT ASYMMETRY
    # --------------------------------------------------------

    result["ankle_asymmetry_deg"] = (
        num(
            result,
            "left_ankle_mean_deg"
        )
        -
        num(
            result,
            "right_ankle_mean_deg"
        )
    ).abs()

    # --------------------------------------------------------
    # TRUNK EXCURSION
    # --------------------------------------------------------

    result["trunk_excursion_deg"] = num(
        result,
        "trunk_range_deg"
    )

    # --------------------------------------------------------
    # MAXIMUM JOINT ASYMMETRY
    # --------------------------------------------------------

    result["maximum_joint_asymmetry_deg"] = result[
        [
            "knee_asymmetry_deg",
            "hip_asymmetry_deg",
            "ankle_asymmetry_deg"
        ]
    ].max(axis=1)

    return result


# ============================================================
# DATASET-DERIVED UPPER-5% FLAGS
# ============================================================

def add_flags(df):

    indicators = [
        "knee_asymmetry_deg",
        "hip_asymmetry_deg",
        "ankle_asymmetry_deg",
        "trunk_excursion_deg"
    ]

    thresholds = {}

    for column in indicators:

        values = (
            pd.to_numeric(
                df[column],
                errors="coerce"
            )
            .dropna()
        )

        if values.empty:

            thresholds[column] = np.nan

            continue

        threshold = np.percentile(
            values,
            95
        )

        thresholds[column] = threshold

        flag_name = (
            column
            .replace("_deg", "")
            + "_upper5pct_flag"
        )

        df[flag_name] = (
            df[column]
            >= threshold
        ).astype(int)

    return df, thresholds


# ============================================================
# PRINT SUMMARY
# ============================================================

def print_summary(
    df,
    thresholds,
    view
):

    print()
    print("=" * 70)
    print(
        f"{view.upper()} — MOVEMENT INDICATOR SUMMARY"
    )
    print("=" * 70)

    print()

    for column, threshold in thresholds.items():

        if np.isnan(threshold):

            print(
                f"{column}: no valid data"
            )

        else:

            print(
                f"{column}: "
                f"95th percentile = "
                f"{threshold:.2f} deg"
            )

    print()

    for column in thresholds:

        flag = (
            column
            .replace("_deg", "")
            + "_upper5pct_flag"
        )

        if flag not in df.columns:
            continue

        count = int(
            df[flag].sum()
        )

        total = len(df)

        percentage = (
            count / total * 100
        )

        print(
            f"{column}: "
            f"{count}/{total} "
            f"({percentage:.1f}%) "
            f"upper-5% observations"
        )


# ============================================================
# SAVE RISK TABLE
# ============================================================

def save_table(
    df,
    view
):

    columns = [
        "unique_rep",
        "participant",
        "rep_id",

        "knee_asymmetry_deg",
        "hip_asymmetry_deg",
        "ankle_asymmetry_deg",

        "trunk_excursion_deg",

        "maximum_joint_asymmetry_deg",

        "knee_asymmetry_upper5pct_flag",
        "hip_asymmetry_upper5pct_flag",
        "ankle_asymmetry_upper5pct_flag",
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
        / f"{view}_risk_indicators.csv"
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
# DISTRIBUTION GRAPH
# ============================================================

def distribution_graph(
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
    )

    if values.empty:

        print(
            f"[WARNING] No valid data for {column}"
        )

        return

    mean = values.mean()

    threshold = np.percentile(
        values,
        95
    )

    fig, ax = plt.subplots(
        figsize=(9, 5.5)
    )

    ax.hist(
        values,
        bins=15,
        alpha=0.75
    )

    ax.axvline(
        mean,
        linewidth=2,
        label=f"Mean = {mean:.2f}°"
    )

    ax.axvline(
        threshold,
        linestyle="--",
        linewidth=2,
        label=f"95th percentile = {threshold:.2f}°"
    )

    ax.set_title(
        title,
        fontsize=15,
        fontweight="bold"
    )

    ax.set_xlabel(
        xlabel
    )

    ax.set_ylabel(
        "Number of repetitions"
    )

    ax.grid(
        True,
        alpha=0.20
    )

    ax.legend(
        frameon=False
    )

    fig.tight_layout()

    png = GRAPH_ROOT / f"{filename}.png"
    pdf = GRAPH_ROOT / f"{filename}.pdf"

    fig.savefig(
        png,
        dpi=300,
        bbox_inches="tight"
    )

    fig.savefig(
        pdf,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(
        f"[DONE] {png.name}"
    )


# ============================================================
# OVERVIEW GRAPH
# ============================================================

def overview_graph(df):

    columns = [
        "knee_asymmetry_deg",
        "hip_asymmetry_deg",
        "ankle_asymmetry_deg",
        "trunk_excursion_deg"
    ]

    labels = [
        "Knee\nasymmetry",
        "Hip\nasymmetry",
        "Ankle\nasymmetry",
        "Trunk\nexcursion"
    ]

    means = []

    for column in columns:

        values = (
            pd.to_numeric(
                df[column],
                errors="coerce"
            )
            .dropna()
        )

        if values.empty:
            means.append(np.nan)
        else:
            means.append(values.mean())

    fig, ax = plt.subplots(
        figsize=(9, 5.5)
    )

    x = np.arange(
        len(labels)
    )

    ax.bar(
        x,
        means
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        labels
    )

    ax.set_ylabel(
        "Mean angular measure (degrees)"
    )

    ax.set_title(
        "Derived Movement Indicators",
        fontsize=15,
        fontweight="bold"
    )

    ax.grid(
        axis="y",
        alpha=0.20
    )

    fig.tight_layout()

    png = (
        GRAPH_ROOT
        / "risk_indicator_overview.png"
    )

    pdf = (
        GRAPH_ROOT
        / "risk_indicator_overview.pdf"
    )

    fig.savefig(
        png,
        dpi=300,
        bbox_inches="tight"
    )

    fig.savefig(
        pdf,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(
        f"[DONE] {png.name}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print(
        "SQUAT DATASET — RISK-RELATED MOVEMENT ANALYSIS"
    )
    print("=" * 70)

    # ========================================================
    # FRONT
    # ========================================================

    front = load_features(
        "front"
    )

    front = calculate_indicators(
        front
    )

    front, front_thresholds = add_flags(
        front
    )

    print_summary(
        front,
        front_thresholds,
        "front"
    )

    save_table(
        front,
        "front"
    )

    # ========================================================
    # SIDE
    # ========================================================

    side = load_features(
        "side"
    )

    side = calculate_indicators(
        side
    )

    side, side_thresholds = add_flags(
        side
    )

    print_summary(
        side,
        side_thresholds,
        "side"
    )

    save_table(
        side,
        "side"
    )

    # ========================================================
    # GRAPHS
    # ========================================================

    print()
    print("=" * 70)
    print(
        "GENERATING MOVEMENT INDICATOR GRAPHS"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # Front knee
    # --------------------------------------------------------

    distribution_graph(
        front,
        "knee_asymmetry_deg",
        "Distribution of Knee Left-Right Asymmetry",
        "Absolute knee angle difference (degrees)",
        "front_knee_asymmetry_distribution"
    )

    # --------------------------------------------------------
    # Front hip
    # --------------------------------------------------------

    distribution_graph(
        front,
        "hip_asymmetry_deg",
        "Distribution of Hip Left-Right Asymmetry",
        "Absolute hip angle difference (degrees)",
        "front_hip_asymmetry_distribution"
    )

    # --------------------------------------------------------
    # Front ankle
    # --------------------------------------------------------

    distribution_graph(
        front,
        "ankle_asymmetry_deg",
        "Distribution of Ankle Left-Right Asymmetry",
        "Absolute ankle angle difference (degrees)",
        "front_ankle_asymmetry_distribution"
    )

    # --------------------------------------------------------
    # Side trunk
    # --------------------------------------------------------

    distribution_graph(
        side,
        "trunk_excursion_deg",
        "Distribution of Trunk Angular Excursion",
        "Trunk angular excursion (degrees)",
        "trunk_excursion_distribution"
    )

    # --------------------------------------------------------
    # Overview
    # --------------------------------------------------------

    overview_graph(
        front
    )

    # ========================================================
    # COMPLETE
    # ========================================================

    print()
    print("=" * 70)
    print(
        "RISK-RELATED MOVEMENT ANALYSIS COMPLETE"
    )
    print("=" * 70)

    print()
    print(
        "Tables:"
    )

    print(
        RISK_ROOT
    )

    print()
    print(
        "Graphs:"
    )

    print(
        GRAPH_ROOT
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "These are pose-derived movement indicators."
    )

    print(
        "They are not clinical injury diagnoses."
    )

    print(
        "Upper-5% flags are dataset-derived statistical flags,"
    )

    print(
        "not medical injury thresholds."
    )

    print()


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()