"""
07_risk_patterns.py

SQUAT DATASET — RISK-RELATED MOVEMENT INDICATORS
=================================================

This module derives descriptive movement indicators from the
pose-derived squat features.

IMPORTANT:
These are NOT clinical injury diagnoses and NOT clinical
injury probabilities.

The dataset contains kinematic measurements rather than
clinically confirmed injury outcomes.

Indicators generated:

1. Knee left-right asymmetry
2. Hip left-right asymmetry
3. Ankle left-right asymmetry
4. Trunk excursion
5. Joint range of motion
6. Extreme movement flag

Thresholds for the extreme-movement flags are DATASET-DERIVED
(95th percentile), not medical thresholds.

Outputs:

data/processed/risk/
    front_risk_indicators.csv
    side_risk_indicators.csv

outputs/risk_graphs/
    front_knee_asymmetry_distribution.png
    front_hip_asymmetry_distribution.png
    front_ankle_asymmetry_distribution.png
    trunk_excursion_distribution.png
    risk_indicator_overview.png
"""


from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

FEATURE_ROOT = (
    ROOT
    / "data"
    / "processed"
    / "features"
)

RISK_ROOT = (
    ROOT
    / "data"
    / "processed"
    / "risk"
)

GRAPH_ROOT = (
    ROOT
    / "outputs"
    / "risk_graphs"
)

RISK_ROOT.mkdir(
    parents=True,
    exist_ok=True
)

GRAPH_ROOT.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# FILE LOADING
# ============================================================

def load_features(view):

    path = (
        FEATURE_ROOT
        / f"{view}_features.csv"
    )

    print()
    print("=" * 70)
    print(f"LOADING {view.upper()} FEATURES")
    print("=" * 70)

    if not path.exists():

        raise FileNotFoundError(
            f"Feature file not found:\n{path}"
        )

    df = pd.read_csv(path)

    print(
        f"Rows: {len(df)}"
    )

    print(
        f"Columns: {len(df.columns)}"
    )

    return df


# ============================================================
# SAFE NUMERIC CONVERSION
# ============================================================

def numeric(
    df,
    column
):

    if column not in df.columns:

        return pd.Series(
            np.nan,
            index=df.index
        )

    return pd.to_numeric(
        df[column],
        errors="coerce"
    )


# ============================================================
# FEATURE CALCULATION
# ============================================================

def calculate_indicators(df):

    result = df.copy()

    # ========================================================
    # KNEE ASYMMETRY
    # ========================================================

    result["knee_asymmetry_deg"] = (
        numeric(
            result,
            "left_knee_angle_deg_mean"
        )
        -
        numeric(
            result,
            "right_knee_angle_deg_mean"
        )
    ).abs()

    # ========================================================
    # HIP ASYMMETRY
    # ========================================================

    result["hip_asymmetry_deg"] = (
        numeric(
            result,
            "left_hip_angle_deg_mean"
        )
        -
        numeric(
            result,
            "right_hip_angle_deg_mean"
        )
    ).abs()

    # ========================================================
    # ANKLE ASYMMETRY
    # ========================================================

    result["ankle_asymmetry_deg"] = (
        numeric(
            result,
            "left_ankle_angle_deg_mean"
        )
        -
        numeric(
            result,
            "right_ankle_angle_deg_mean"
        )
    ).abs()

    # ========================================================
    # TRUNK EXCURSION
    #
    # Range of trunk angle during the squat.
    # ========================================================

    result["trunk_excursion_deg"] = numeric(
        result,
        "trunk_angle_deg_range_deg"
    )

    # ========================================================
    # LOWER-LIMB RANGE OF MOTION
    # ========================================================

    result["knee_rom_deg"] = (
        numeric(
            result,
            "left_knee_angle_deg_range_deg"
        )
        +
        numeric(
            result,
            "right_knee_angle_deg_range_deg"
        )
    ) / 2

    result["hip_rom_deg"] = (
        numeric(
            result,
            "left_hip_angle_deg_range_deg"
        )
        +
        numeric(
            result,
            "right_hip_angle_deg_range_deg"
        )
    ) / 2

    result["ankle_rom_deg"] = (
        numeric(
            result,
            "left_ankle_angle_deg_range_deg"
        )
        +
        numeric(
            result,
            "right_ankle_angle_deg_range_deg"
        )
    ) / 2

    # ========================================================
    # MAXIMUM ASYMMETRY
    # ========================================================

    result["maximum_joint_asymmetry_deg"] = result[
        [
            "knee_asymmetry_deg",
            "hip_asymmetry_deg",
            "ankle_asymmetry_deg"
        ]
    ].max(axis=1)

    return result


# ============================================================
# DATASET-DERIVED EXTREME FLAGS
# ============================================================

def add_dataset_flags(df):

    # --------------------------------------------------------
    # We deliberately use the 95th percentile of THIS dataset.
    #
    # This does not mean:
    # "95th percentile = injury".
    #
    # It means:
    # "movement lies in the upper 5% of observed values."
    # --------------------------------------------------------

    indicators = [
        "knee_asymmetry_deg",
        "hip_asymmetry_deg",
        "ankle_asymmetry_deg",
        "trunk_excursion_deg"
    ]

    thresholds = {}

    for column in indicators:

        values = pd.to_numeric(
            df[column],
            errors="coerce"
        ).dropna()

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
# SUMMARY
# ============================================================

def print_summary(
    df,
    thresholds,
    view
):

    print()
    print("=" * 70)
    print(
        f"{view.upper()} — RISK-RELATED MOVEMENT SUMMARY"
    )
    print("=" * 70)

    print()

    for column, threshold in thresholds.items():

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

        if flag in df.columns:

            count = int(
                df[flag].sum()
            )

            total = len(df)

            percentage = (
                count
                /
                total
                *
                100
            )

            print(
                f"{column}: "
                f"{count}/{total} "
                f"({percentage:.1f}%) "
                f"in dataset upper 5%"
            )


# ============================================================
# SAVE TABLE
# ============================================================

def save_risk_table(
    df,
    view
):

    output = (
        RISK_ROOT
        / f"{view}_risk_indicators.csv"
    )

    # --------------------------------------------------------
    # Keep publication-relevant columns.
    # --------------------------------------------------------

    preferred = [
        "participant",
        "rep_id",
        "unique_rep",

        "knee_asymmetry_deg",
        "hip_asymmetry_deg",
        "ankle_asymmetry_deg",

        "trunk_excursion_deg",

        "knee_rom_deg",
        "hip_rom_deg",
        "ankle_rom_deg",

        "maximum_joint_asymmetry_deg",

        "knee_asymmetry_upper5pct_flag",
        "hip_asymmetry_upper5pct_flag",
        "ankle_asymmetry_upper5pct_flag",
        "trunk_excursion_upper5pct_flag",
    ]

    columns = [
        c
        for c in preferred
        if c in df.columns
    ]

    output_df = df[
        columns
    ].copy()

    output_df.to_csv(
        output,
        index=False
    )

    print()
    print(
        f"[DONE] Saved: {output}"
    )

    print(
        f"Rows: {len(output_df)}"
    )

    print(
        f"Columns: {len(output_df.columns)}"
    )


# ============================================================
# DISTRIBUTION GRAPH
# ============================================================

def plot_distribution(
    df,
    column,
    title,
    filename
):

    values = pd.to_numeric(
        df[column],
        errors="coerce"
    ).dropna()

    if values.empty:

        print(
            f"[WARNING] No data for {column}"
        )

        return

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
        values.mean(),
        linestyle="-",
        linewidth=2,
        label=f"Mean = {values.mean():.2f}°"
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
        "Angle difference (degrees)"
    )

    ax.set_ylabel(
        "Number of repetitions"
    )

    ax.grid(
        True,
        alpha=0.2
    )

    ax.legend(
        frameon=False
    )

    fig.tight_layout()

    png = (
        GRAPH_ROOT
        / f"{filename}.png"
    )

    pdf = (
        GRAPH_ROOT
        / f"{filename}.pdf"
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
# TRUNK GRAPH
# ============================================================

def plot_trunk_distribution(
    df
):

    values = pd.to_numeric(
        df["trunk_excursion_deg"],
        errors="coerce"
    ).dropna()

    if values.empty:

        return

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
        values.mean(),
        linewidth=2,
        label=f"Mean = {values.mean():.2f}°"
    )

    ax.axvline(
        threshold,
        linestyle="--",
        linewidth=2,
        label=f"95th percentile = {threshold:.2f}°"
    )

    ax.set_title(
        "Trunk Angular Excursion Across Repetitions",
        fontsize=15,
        fontweight="bold"
    )

    ax.set_xlabel(
        "Trunk angular excursion (degrees)"
    )

    ax.set_ylabel(
        "Number of repetitions"
    )

    ax.grid(
        True,
        alpha=0.2
    )

    ax.legend(
        frameon=False
    )

    fig.tight_layout()

    png = (
        GRAPH_ROOT
        / "trunk_excursion_distribution.png"
    )

    pdf = (
        GRAPH_ROOT
        / "trunk_excursion_distribution.pdf"
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
# OVERVIEW GRAPH
# ============================================================

def plot_overview(
    df
):

    columns = [
        "knee_asymmetry_deg",
        "hip_asymmetry_deg",
        "ankle_asymmetry_deg",
        "trunk_excursion_deg"
    ]

    labels = [
        "Knee asymmetry",
        "Hip asymmetry",
        "Ankle asymmetry",
        "Trunk excursion"
    ]

    means = []

    for column in columns:

        if column not in df.columns:

            means.append(
                np.nan
            )

            continue

        values = pd.to_numeric(
            df[column],
            errors="coerce"
        ).dropna()

        if values.empty:

            means.append(
                np.nan
            )

        else:

            means.append(
                values.mean()
            )

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

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        labels,
        rotation=15
    )

    ax.set_ylabel(
        "Mean angular measure (degrees)"
    )

    ax.set_title(
        "Summary of Derived Movement Indicators",
        fontsize=15,
        fontweight="bold"
    )

    ax.grid(
        axis="y",
        alpha=0.2
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
# PROCESS ONE VIEW
# ============================================================

def process_view(view):

    df = load_features(
        view
    )

    df = calculate_indicators(
        df
    )

    df, thresholds = add_dataset_flags(
        df
    )

    print_summary(
        df,
        thresholds,
        view
    )

    save_risk_table(
        df,
        view
    )

    return df


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print(
        "SQUAT DATASET — RISK-RELATED MOVEMENT INDICATORS"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # FRONT VIEW
    # --------------------------------------------------------

    front = process_view(
        "front"
    )

    # --------------------------------------------------------
    # SIDE VIEW
    # --------------------------------------------------------

    side = process_view(
        "side"
    )

    # --------------------------------------------------------
    # FRONT VIEW GRAPHS
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "GENERATING FRONT-VIEW INDICATOR GRAPHS"
    )
    print("=" * 70)

    plot_distribution(
        front,
        "knee_asymmetry_deg",
        "Distribution of Knee Left-Right Asymmetry",
        "front_knee_asymmetry_distribution"
    )

    plot_distribution(
        front,
        "hip_asymmetry_deg",
        "Distribution of Hip Left-Right Asymmetry",
        "front_hip_asymmetry_distribution"
    )

    plot_distribution(
        front,
        "ankle_asymmetry_deg",
        "Distribution of Ankle Left-Right Asymmetry",
        "front_ankle_asymmetry_distribution"
    )

    # --------------------------------------------------------
    # SIDE VIEW TRUNK
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "GENERATING SIDE-VIEW TRUNK GRAPH"
    )
    print("=" * 70)

    plot_trunk_distribution(
        side
    )

    # --------------------------------------------------------
    # COMBINED OVERVIEW
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "GENERATING INDICATOR OVERVIEW"
    )
    print("=" * 70)

    plot_overview(
        front
    )

    # --------------------------------------------------------
    # COMPLETE
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "RISK-RELATED INDICATOR GENERATION COMPLETE"
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
        "These indicators describe observed movement patterns."
    )

    print(
        "They do NOT represent clinical injury diagnosis."
    )

    print(
        "The upper-5% flags are dataset-derived statistical flags,"
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