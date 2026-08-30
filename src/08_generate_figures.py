"""
08_generate_figures.py

SQUAT DATASET PUBLICATION PIPELINE
==================================

Purpose
-------
Generate publication-oriented biomechanical figures from the
temporally normalized squat dataset.

IMPORTANT SCIENTIFIC DESIGN
----------------------------
The front and side cameras observe different anatomical planes.

SIDE VIEW
---------
Used primarily for sagittal-plane movement:
    - Knee flexion/extension
    - Hip flexion/extension
    - Ankle angle
    - Trunk inclination

FRONT VIEW
----------
Used primarily to examine left-right movement patterns:
    - Left knee trajectory
    - Right knee trajectory
    - Left-right knee asymmetry
    - Left hip trajectory
    - Right hip trajectory
    - Left-right hip asymmetry
    - Left/right ankle trajectories
    - Trunk trajectory

We therefore DO NOT directly interpret a front-view projected
knee angle as equivalent to a side-view sagittal knee angle.

INPUT
-----
data/processed/normalized/front/all_normalized_repetitions.csv
data/processed/normalized/side/all_normalized_repetitions.csv

OUTPUT
------
outputs/angle_graphs/
outputs/comparison_graph/

FIGURES
-------
SIDE VIEW
    1. Knee trajectory
    2. Hip trajectory
    3. Ankle trajectory
    4. Trunk trajectory

FRONT VIEW
    5. Knee left vs right
    6. Hip left vs right
    7. Ankle left vs right
    8. Trunk trajectory
    9. Knee asymmetry
   10. Hip asymmetry
   11. Ankle asymmetry

COMPARISON
   12. Side-view knee vs hip
   13. Side-view lower-limb angles
   14. Front-view asymmetry overview

Each figure contains:
    - Individual repetitions where appropriate
    - Mean trajectory
    - ±1 standard deviation
    - 0–100% normalized squat cycle

These are kinematic/pose-derived measurements and are NOT
clinical injury diagnoses.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

NORMALIZED_DIR = (
    ROOT
    / "data"
    / "processed"
    / "normalized"
)

OUTPUT_DIR = (
    ROOT
    / "outputs"
)

ANGLE_OUTPUT_DIR = (
    OUTPUT_DIR
    / "angle_graphs"
)

COMPARISON_OUTPUT_DIR = (
    OUTPUT_DIR
    / "comparison_graph"
)

ANGLE_OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

COMPARISON_OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. GENERAL SETTINGS
# ============================================================

DPI = 300

FIG_WIDTH = 9

FIG_HEIGHT = 6

CYCLE_MIN = 0

CYCLE_MAX = 100


# ============================================================
# 3. DATA COLUMNS
# ============================================================

SIDE_SIGNALS = {

    "knee": "side_knee",

    "hip": "side_hip",

    "ankle": "side_ankle",

    "trunk": "side_trunk",
}


FRONT_PAIRS = {

    "knee": (
        "left_knee_angle_deg",
        "right_knee_angle_deg",
    ),

    "hip": (
        "left_hip_angle_deg",
        "right_hip_angle_deg",
    ),

    "ankle": (
        "left_ankle_angle_deg",
        "right_ankle_angle_deg",
    ),
}


# ============================================================
# 4. LOAD DATA
# ============================================================

def load_view(view):

    file_path = (
        NORMALIZED_DIR
        / view
        / "all_normalized_repetitions.csv"
    )

    if not file_path.exists():

        raise FileNotFoundError(
            f"Dataset not found:\n{file_path}"
        )

    df = pd.read_csv(
        file_path
    )

    if df.empty:

        raise ValueError(
            f"Dataset is empty:\n{file_path}"
        )

    return df


# ============================================================
# 5. PREPARE DATA
# ============================================================

def prepare_data(df):

    df = df.copy()

    # --------------------------------------------------------
    # rep_id restarts for each participant.
    #
    # participant + rep_id therefore defines one unique
    # squat repetition.
    # --------------------------------------------------------

    df["unique_rep"] = (
        df["participant"].astype(str)
        + "_rep_"
        + df["rep_id"].astype(str)
    )

    df["cycle_percent"] = pd.to_numeric(
        df["cycle_percent"],
        errors="coerce"
    )

    return df


# ============================================================
# 6. STATISTICS FOR ONE SIGNAL
# ============================================================

def calculate_statistics(
    df,
    signal_column
):

    if signal_column not in df.columns:

        return pd.DataFrame()

    data = df[
        [
            "unique_rep",
            "cycle_percent",
            signal_column,
        ]
    ].copy()

    data[signal_column] = pd.to_numeric(
        data[signal_column],
        errors="coerce"
    )

    data = data.dropna(
        subset=[
            "cycle_percent",
            signal_column,
        ]
    )

    if data.empty:

        return pd.DataFrame()

    grouped = (
        data
        .groupby(
            "cycle_percent"
        )[signal_column]
    )

    result = pd.DataFrame({

        "cycle_percent":
            grouped.mean().index,

        "mean":
            grouped.mean().values,

        "std":
            grouped.std(
                ddof=1
            ).fillna(0).values,

        "n":
            grouped.count().values,
    })

    result["upper"] = (
        result["mean"]
        +
        result["std"]
    )

    result["lower"] = (
        result["mean"]
        -
        result["std"]
    )

    return result


# ============================================================
# 7. UNIQUE REPETITIONS
# ============================================================

def get_repetitions(
    df,
    signal_column
):

    if signal_column not in df.columns:

        return []

    valid = df[
        [
            "unique_rep",
            "cycle_percent",
            signal_column,
        ]
    ].copy()

    valid[signal_column] = pd.to_numeric(
        valid[signal_column],
        errors="coerce"
    )

    valid = valid.dropna(
        subset=[
            "cycle_percent",
            signal_column,
        ]
    )

    return (
        valid[
            "unique_rep"
        ]
        .drop_duplicates()
        .tolist()
    )


# ============================================================
# 8. BASE FIGURE STYLE
# ============================================================

def prepare_axis(
    ax,
    title,
    ylabel
):

    ax.set_title(
        title,
        fontsize=14,
        pad=12
    )

    ax.set_xlabel(
        "Normalized Squat Cycle (%)",
        fontsize=12
    )

    ax.set_ylabel(
        ylabel,
        fontsize=12
    )

    ax.set_xlim(
        CYCLE_MIN,
        CYCLE_MAX
    )

    ax.grid(
        True,
        alpha=0.22,
        linewidth=0.8
    )

    ax.tick_params(
        labelsize=10
    )


# ============================================================
# 9. SIDE VIEW — SINGLE SIGNAL
# ============================================================

def plot_side_signal(
    df,
    signal_column,
    title,
    filename
):

    statistics = calculate_statistics(
        df,
        signal_column
    )

    if statistics.empty:

        print(
            f"[WARNING] No data for {signal_column}"
        )

        return

    fig, ax = plt.subplots(
        figsize=(
            FIG_WIDTH,
            FIG_HEIGHT
        )
    )

    # --------------------------------------------------------
    # Individual repetitions
    # --------------------------------------------------------

    repetitions = get_repetitions(
        df,
        signal_column
    )

    for rep in repetitions:

        rep_df = df[
            df["unique_rep"]
            == rep
        ].copy()

        rep_df[signal_column] = pd.to_numeric(
            rep_df[signal_column],
            errors="coerce"
        )

        rep_df = rep_df.dropna(
            subset=[
                "cycle_percent",
                signal_column,
            ]
        )

        ax.plot(
            rep_df[
                "cycle_percent"
            ],
            rep_df[
                signal_column
            ],
            linewidth=0.7,
            alpha=0.15
        )

    # --------------------------------------------------------
    # Mean
    # --------------------------------------------------------

    ax.plot(
        statistics[
            "cycle_percent"
        ],
        statistics[
            "mean"
        ],
        linewidth=2.7,
        label="Mean"
    )

    # --------------------------------------------------------
    # ±1 SD
    # --------------------------------------------------------

    ax.fill_between(

        statistics[
            "cycle_percent"
        ],

        statistics[
            "lower"
        ],

        statistics[
            "upper"
        ],

        alpha=0.18,

        label="±1 SD"
    )

    # --------------------------------------------------------
    # Bottom marker
    # --------------------------------------------------------

    ax.axvline(
        50,
        linestyle="--",
        linewidth=1.0,
        alpha=0.65
    )

    ax.text(
        50,
        0.97,
        "Approx. bottom",
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=9
    )

    prepare_axis(
        ax,
        title,
        "Angle (degrees)"
    )

    ax.legend(
        frameon=False
    )

    fig.tight_layout()

    png_path = (
        ANGLE_OUTPUT_DIR
        / f"{filename}.png"
    )

    pdf_path = (
        ANGLE_OUTPUT_DIR
        / f"{filename}.pdf"
    )

    fig.savefig(
        png_path,
        dpi=DPI,
        bbox_inches="tight"
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight"
    )

    plt.close(
        fig
    )

    print(
        f"[DONE] {png_path.name}"
    )


# ============================================================
# 10. FRONT VIEW — LEFT VS RIGHT
# ============================================================

def plot_front_left_right(
    df,
    left_column,
    right_column,
    joint_name
):

    left_stats = calculate_statistics(
        df,
        left_column
    )

    right_stats = calculate_statistics(
        df,
        right_column
    )

    if left_stats.empty or right_stats.empty:

        print(
            f"[WARNING] Missing front-view data "
            f"for {joint_name}"
        )

        return

    fig, ax = plt.subplots(
        figsize=(
            FIG_WIDTH,
            FIG_HEIGHT
        )
    )

    # --------------------------------------------------------
    # Left
    # --------------------------------------------------------

    ax.plot(
        left_stats[
            "cycle_percent"
        ],
        left_stats[
            "mean"
        ],
        linewidth=2.5,
        label="Left"
    )

    ax.fill_between(

        left_stats[
            "cycle_percent"
        ],

        left_stats[
            "lower"
        ],

        left_stats[
            "upper"
        ],

        alpha=0.12
    )

    # --------------------------------------------------------
    # Right
    # --------------------------------------------------------

    ax.plot(
        right_stats[
            "cycle_percent"
        ],
        right_stats[
            "mean"
        ],
        linewidth=2.5,
        linestyle="--",
        label="Right"
    )

    ax.fill_between(

        right_stats[
            "cycle_percent"
        ],

        right_stats[
            "lower"
        ],

        right_stats[
            "upper"
        ],

        alpha=0.12
    )

    # --------------------------------------------------------
    # Bottom
    # --------------------------------------------------------

    ax.axvline(
        50,
        linestyle=":",
        linewidth=1.0,
        alpha=0.65
    )

    prepare_axis(
        ax,
        f"Front View — {joint_name} Left–Right Trajectory",
        "Projected angle (degrees)"
    )

    ax.legend(
        frameon=False
    )

    fig.tight_layout()

    png_path = (
        ANGLE_OUTPUT_DIR
        / f"front_{joint_name.lower()}_left_right.png"
    )

    pdf_path = (
        ANGLE_OUTPUT_DIR
        / f"front_{joint_name.lower()}_left_right.pdf"
    )

    fig.savefig(
        png_path,
        dpi=DPI,
        bbox_inches="tight"
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight"
    )

    plt.close(
        fig
    )

    print(
        f"[DONE] {png_path.name}"
    )


# ============================================================
# 11. FRONT VIEW — ASYMMETRY
# ============================================================

def calculate_asymmetry(
    df,
    left_column,
    right_column
):

    data = df[
        [
            "unique_rep",
            "cycle_percent",
            left_column,
            right_column,
        ]
    ].copy()

    data[left_column] = pd.to_numeric(
        data[left_column],
        errors="coerce"
    )

    data[right_column] = pd.to_numeric(
        data[right_column],
        errors="coerce"
    )

    data["asymmetry_deg"] = (
        data[left_column]
        -
        data[right_column]
    ).abs()

    return data


def plot_front_asymmetry(
    df,
    left_column,
    right_column,
    joint_name
):

    data = calculate_asymmetry(
        df,
        left_column,
        right_column
    )

    data = data.dropna(
        subset=[
            "cycle_percent",
            "asymmetry_deg",
        ]
    )

    if data.empty:

        print(
            f"[WARNING] No asymmetry data "
            f"for {joint_name}"
        )

        return

    grouped = (
        data
        .groupby(
            "cycle_percent"
        )[
            "asymmetry_deg"
        ]
    )

    statistics = pd.DataFrame({

        "cycle_percent":
            grouped.mean().index,

        "mean":
            grouped.mean().values,

        "std":
            grouped.std(
                ddof=1
            ).fillna(0).values,
    })

    statistics["upper"] = (
        statistics["mean"]
        +
        statistics["std"]
    )

    statistics["lower"] = (
        statistics["mean"]
        -
        statistics["std"]
    )

    fig, ax = plt.subplots(
        figsize=(
            FIG_WIDTH,
            FIG_HEIGHT
        )
    )

    # --------------------------------------------------------
    # Individual repetition asymmetry curves
    # --------------------------------------------------------

    repetitions = (
        data[
            "unique_rep"
        ]
        .drop_duplicates()
        .tolist()
    )

    for rep in repetitions:

        rep_df = data[
            data["unique_rep"]
            == rep
        ]

        ax.plot(
            rep_df[
                "cycle_percent"
            ],
            rep_df[
                "asymmetry_deg"
            ],
            linewidth=0.7,
            alpha=0.14
        )

    # --------------------------------------------------------
    # Mean asymmetry
    # --------------------------------------------------------

    ax.plot(
        statistics[
            "cycle_percent"
        ],
        statistics[
            "mean"
        ],
        linewidth=2.7,
        label="Mean absolute difference"
    )

    # --------------------------------------------------------
    # SD
    # --------------------------------------------------------

    ax.fill_between(

        statistics[
            "cycle_percent"
        ],

        statistics[
            "lower"
        ],

        statistics[
            "upper"
        ],

        alpha=0.18,

        label="±1 SD"
    )

    ax.axvline(
        50,
        linestyle=":",
        linewidth=1.0,
        alpha=0.65
    )

    prepare_axis(
        ax,
        f"Front View — {joint_name} Left–Right Asymmetry",
        "Absolute angle difference (degrees)"
    )

    ax.legend(
        frameon=False
    )

    fig.tight_layout()

    png_path = (
        ANGLE_OUTPUT_DIR
        / f"front_{joint_name.lower()}_asymmetry.png"
    )

    pdf_path = (
        ANGLE_OUTPUT_DIR
        / f"front_{joint_name.lower()}_asymmetry.pdf"
    )

    fig.savefig(
        png_path,
        dpi=DPI,
        bbox_inches="tight"
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight"
    )

    plt.close(
        fig
    )

    print(
        f"[DONE] {png_path.name}"
    )


# ============================================================
# 12. SIDE VIEW — LOWER LIMB COMPARISON
# ============================================================

def plot_side_lower_limb(
    df
):

    signals = [
        (
            "left_knee_angle_deg",
            "Knee"
        ),

        (
            "left_hip_angle_deg",
            "Hip"
        ),

        (
            "left_ankle_angle_deg",
            "Ankle"
        ),
    ]

    fig, ax = plt.subplots(
        figsize=(
            FIG_WIDTH,
            FIG_HEIGHT
        )
    )

    for column, label in signals:

        if column not in df.columns:

            continue

        statistics = calculate_statistics(
            df,
            column
        )

        if statistics.empty:

            continue

        ax.plot(
            statistics[
                "cycle_percent"
            ],
            statistics[
                "mean"
            ],
            linewidth=2.2,
            label=label
        )

    ax.axvline(
        50,
        linestyle=":",
        linewidth=1.0,
        alpha=0.65
    )

    prepare_axis(
        ax,
        "Side View — Lower-Limb Joint-Angle Trajectories",
        "Angle (degrees)"
    )

    ax.legend(
        frameon=False
    )

    fig.tight_layout()

    png_path = (
        COMPARISON_OUTPUT_DIR
        / "side_lower_limb_trajectories.png"
    )

    pdf_path = (
        COMPARISON_OUTPUT_DIR
        / "side_lower_limb_trajectories.pdf"
    )

    fig.savefig(
        png_path,
        dpi=DPI,
        bbox_inches="tight"
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight"
    )

    plt.close(
        fig
    )

    print(
        f"[DONE] {png_path.name}"
    )


# ============================================================
# 13. SIDE VIEW — KNEE AND HIP
# ============================================================

def plot_side_knee_hip(
    df
):

    knee = calculate_statistics(
        df,
        "left_knee_angle_deg"
    )

    hip = calculate_statistics(
        df,
        "left_hip_angle_deg"
    )

    if knee.empty or hip.empty:

        return

    fig, ax = plt.subplots(
        figsize=(
            FIG_WIDTH,
            FIG_HEIGHT
        )
    )

    ax.plot(
        knee[
            "cycle_percent"
        ],
        knee[
            "mean"
        ],
        linewidth=2.5,
        label="Knee"
    )

    ax.plot(
        hip[
            "cycle_percent"
        ],
        hip[
            "mean"
        ],
        linewidth=2.5,
        label="Hip"
    )

    ax.axvline(
        50,
        linestyle=":",
        linewidth=1.0,
        alpha=0.65
    )

    prepare_axis(
        ax,
        "Side View — Knee and Hip Kinematics",
        "Angle (degrees)"
    )

    ax.legend(
        frameon=False
    )

    fig.tight_layout()

    png_path = (
        COMPARISON_OUTPUT_DIR
        / "side_knee_hip_trajectories.png"
    )

    pdf_path = (
        COMPARISON_OUTPUT_DIR
        / "side_knee_hip_trajectories.pdf"
    )

    fig.savefig(
        png_path,
        dpi=DPI,
        bbox_inches="tight"
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight"
    )

    plt.close(
        fig
    )

    print(
        f"[DONE] {png_path.name}"
    )


# ============================================================
# 14. FRONT VIEW — ASYMMETRY OVERVIEW
# ============================================================

def plot_front_asymmetry_overview(
    df
):

    joints = [

        (
            "left_knee_angle_deg",
            "right_knee_angle_deg",
            "Knee"
        ),

        (
            "left_hip_angle_deg",
            "right_hip_angle_deg",
            "Hip"
        ),

        (
            "left_ankle_angle_deg",
            "right_ankle_angle_deg",
            "Ankle"
        ),
    ]

    fig, ax = plt.subplots(
        figsize=(
            FIG_WIDTH,
            FIG_HEIGHT
        )
    )

    for left, right, label in joints:

        data = calculate_asymmetry(
            df,
            left,
            right
        )

        data = data.dropna(
            subset=[
                "cycle_percent",
                "asymmetry_deg",
            ]
        )

        if data.empty:

            continue

        grouped = (
            data
            .groupby(
                "cycle_percent"
            )[
                "asymmetry_deg"
            ]
            .mean()
        )

        ax.plot(
            grouped.index,
            grouped.values,
            linewidth=2.2,
            label=label
        )

    ax.axvline(
        50,
        linestyle=":",
        linewidth=1.0,
        alpha=0.65
    )

    prepare_axis(
        ax,
        "Front View — Left–Right Asymmetry Across Squat Cycle",
        "Absolute angle difference (degrees)"
    )

    ax.legend(
        frameon=False
    )

    fig.tight_layout()

    png_path = (
        COMPARISON_OUTPUT_DIR
        / "front_asymmetry_overview.png"
    )

    pdf_path = (
        COMPARISON_OUTPUT_DIR
        / "front_asymmetry_overview.pdf"
    )

    fig.savefig(
        png_path,
        dpi=DPI,
        bbox_inches="tight"
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight"
    )

    plt.close(
        fig
    )

    print(
        f"[DONE] {png_path.name}"
    )


# ============================================================
# 15. PRINT DATASET SUMMARY
# ============================================================

def print_summary(
    front_df,
    side_df
):

    print()
    print("=" * 70)

    print(
        "DATASET SUMMARY"
    )

    print("=" * 70)

    for name, df in [
        ("Front", front_df),
        ("Side", side_df),
    ]:

        participants = (
            df[
                "participant"
            ]
            .nunique()
        )

        repetitions = (
            df[
                "unique_rep"
            ]
            .nunique()
        )

        points = len(
            df
        )

        print()
        print(
            f"{name} view"
        )

        print(
            f"Participants: "
            f"{participants}"
        )

        print(
            f"Repetitions: "
            f"{repetitions}"
        )

        print(
            f"Normalized points: "
            f"{points}"
        )


# ============================================================
# 16. MAIN
# ============================================================

def main():

    print()
    print("=" * 70)

    print(
        "SQUAT DATASET — REVISED PUBLICATION FIGURES"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    front_df = load_view(
        "front"
    )

    side_df = load_view(
        "side"
    )

    # --------------------------------------------------------
    # Prepare
    # --------------------------------------------------------

    front_df = prepare_data(
        front_df
    )

    side_df = prepare_data(
        side_df
    )

    print_summary(
        front_df,
        side_df
    )

    # ========================================================
    # SIDE VIEW FIGURES
    # ========================================================

    print()
    print("=" * 70)

    print(
        "SIDE VIEW — SAGITTAL KINEMATICS"
    )

    print("=" * 70)

    plot_side_signal(
        side_df,
        "left_knee_angle_deg",
        "Side View — Knee Flexion/Extension Trajectory",
        "side_knee_trajectory"
    )

    plot_side_signal(
        side_df,
        "left_hip_angle_deg",
        "Side View — Hip Flexion/Extension Trajectory",
        "side_hip_trajectory"
    )

    plot_side_signal(
        side_df,
        "left_ankle_angle_deg",
        "Side View — Ankle Angle Trajectory",
        "side_ankle_trajectory"
    )

    plot_side_signal(
        side_df,
        "trunk_angle_deg",
        "Side View — Trunk Angle Trajectory",
        "side_trunk_trajectory"
    )

    # ========================================================
    # FRONT VIEW FIGURES
    # ========================================================

    print()
    print("=" * 70)

    print(
        "FRONT VIEW — LEFT/RIGHT MOVEMENT"
    )

    print("=" * 70)

    plot_front_left_right(
        front_df,
        "left_knee_angle_deg",
        "right_knee_angle_deg",
        "Knee"
    )

    plot_front_left_right(
        front_df,
        "left_hip_angle_deg",
        "right_hip_angle_deg",
        "Hip"
    )

    plot_front_left_right(
        front_df,
        "left_ankle_angle_deg",
        "right_ankle_angle_deg",
        "Ankle"
    )

    plot_side_signal(
        front_df,
        "trunk_angle_deg",
        "Front View — Trunk Angle Trajectory",
        "front_trunk_trajectory"
    )

    # ========================================================
    # FRONT VIEW ASYMMETRY
    # ========================================================

    print()
    print("=" * 70)

    print(
        "FRONT VIEW — ASYMMETRY"
    )

    print("=" * 70)

    plot_front_asymmetry(
        front_df,
        "left_knee_angle_deg",
        "right_knee_angle_deg",
        "Knee"
    )

    plot_front_asymmetry(
        front_df,
        "left_hip_angle_deg",
        "right_hip_angle_deg",
        "Hip"
    )

    plot_front_asymmetry(
        front_df,
        "left_ankle_angle_deg",
        "right_ankle_angle_deg",
        "Ankle"
    )

    # ========================================================
    # COMPARISON FIGURES
    # ========================================================

    print()
    print("=" * 70)

    print(
        "COMPARISON FIGURES"
    )

    print("=" * 70)

    plot_side_lower_limb(
        side_df
    )

    plot_side_knee_hip(
        side_df
    )

    plot_front_asymmetry_overview(
        front_df
    )

    # ========================================================
    # FINISH
    # ========================================================

    print()
    print("=" * 70)

    print(
        "REVISED FIGURE GENERATION COMPLETE"
    )

    print("=" * 70)

    print()
    print(
        "Angle graphs:"
    )

    print(
        ANGLE_OUTPUT_DIR
    )

    print()
    print(
        "Comparison graphs:"
    )

    print(
        COMPARISON_OUTPUT_DIR
    )

    print()
    print(
        "Scientific interpretation:"
    )

    print(
        "Side view = sagittal-plane squat kinematics"
    )

    print(
        "Front view = left-right projected movement/asymmetry"
    )

    print(
        "No clinical injury diagnosis is assigned."
    )

    print()


# ============================================================
# PROGRAM ENTRY
# ============================================================

if __name__ == "__main__":

    main()