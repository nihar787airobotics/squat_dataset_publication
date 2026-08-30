"""
08_generate_figures.py

SQUAT DATASET PUBLICATION FIGURE GENERATOR
==========================================

Dataset:
    20 participants
    5 squat repetitions / participant
    100 repetitions / camera view
    101 normalized points / repetition

Views:
    FRONT  -> frontal-view projected joint trajectories
    SIDE   -> sagittal-view squat kinematics

Purpose:
    Generate clean publication-quality figures for the
    squat dataset paper.

IMPORTANT:
    These measurements are pose-derived kinematic measures.
    They are NOT clinical injury diagnoses.

OUTPUT STRUCTURE
----------------

outputs/
|
+-- angle_graphs/
|   |
|   +-- SIDE VIEW
|   |   +-- side_knee_trajectory.png
|   |   +-- side_hip_trajectory.png
|   |   +-- side_ankle_trajectory.png
|   |   +-- side_trunk_trajectory.png
|   |
|   +-- FRONT VIEW
|       +-- front_knee_left_right.png
|       +-- front_hip_left_right.png
|       +-- front_ankle_left_right.png
|       +-- front_trunk_trajectory.png
|       +-- front_knee_asymmetry.png
|       +-- front_hip_asymmetry.png
|       +-- front_ankle_asymmetry.png
|
+-- comparison_graph/
    |
    +-- side_lower_limb_trajectories.png
    +-- front_asymmetry_overview.png

Each figure is saved as:
    PNG  -> 300 DPI
    PDF  -> vector publication format
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

NORMALIZED = ROOT / "data" / "processed" / "normalized"

OUTPUTS = ROOT / "outputs"

ANGLE_GRAPHS = OUTPUTS / "angle_graphs"

COMPARISON_GRAPHS = OUTPUTS / "comparison_graph"


ANGLE_GRAPHS.mkdir(
    parents=True,
    exist_ok=True
)

COMPARISON_GRAPHS.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# PLOT SETTINGS
# ============================================================

DPI = 300

FIGSIZE = (10, 6)

BOTTOM_PERCENT = 50


# ============================================================
# EXPECTED DATA COLUMNS
# ============================================================

REQUIRED_BASE_COLUMNS = [
    "participant",
    "rep_id",
    "cycle_percent",
]


# ============================================================
# LOAD DATA
# ============================================================

def load_dataset(view):

    path = (
        NORMALIZED
        / view
        / "all_normalized_repetitions.csv"
    )

    print()
    print("-" * 70)
    print(f"Loading {view.upper()} dataset")
    print("-" * 70)
    print(path)

    if not path.exists():

        raise FileNotFoundError(
            f"\nDataset not found:\n{path}"
        )

    df = pd.read_csv(path)

    print(
        f"Rows loaded: {len(df)}"
    )

    print(
        f"Columns: {len(df.columns)}"
    )

    missing = [
        c
        for c in REQUIRED_BASE_COLUMNS
        if c not in df.columns
    ]

    if missing:

        raise ValueError(
            f"\nMissing required columns in {view} dataset:\n"
            f"{missing}\n\n"
            f"Available columns:\n"
            f"{list(df.columns)}"
        )

    # --------------------------------------------------------
    # Make sure numeric columns are actually numeric.
    # --------------------------------------------------------

    df["cycle_percent"] = pd.to_numeric(
        df["cycle_percent"],
        errors="coerce"
    )

    # --------------------------------------------------------
    # Unique repetition.
    #
    # rep_id restarts for every participant.
    # --------------------------------------------------------

    df["unique_rep"] = (
        df["participant"].astype(str)
        + "_rep_"
        + df["rep_id"].astype(str)
    )

    return df


# ============================================================
# VERIFY COLUMNS
# ============================================================

def check_column(
    df,
    column
):

    if column not in df.columns:

        print(
            f"[WARNING] Column not found: {column}"
        )

        return False

    return True


# ============================================================
# CALCULATE STATISTICS
# ============================================================

def get_statistics(
    df,
    column
):

    if not check_column(
        df,
        column
    ):

        return None

    temp = df[
        [
            "unique_rep",
            "cycle_percent",
            column,
        ]
    ].copy()

    temp[column] = pd.to_numeric(
        temp[column],
        errors="coerce"
    )

    temp = temp.dropna(
        subset=[
            "cycle_percent",
            column,
        ]
    )

    if temp.empty:

        print(
            f"[WARNING] No usable values for {column}"
        )

        return None

    grouped = (
        temp
        .groupby(
            "cycle_percent"
        )[column]
    )

    stats = pd.DataFrame({

        "cycle":
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

    stats["upper"] = (
        stats["mean"]
        +
        stats["std"]
    )

    stats["lower"] = (
        stats["mean"]
        -
        stats["std"]
    )

    return stats


# ============================================================
# SAVE FIGURE
# ============================================================

def save_figure(
    fig,
    output_directory,
    filename
):

    png_path = (
        output_directory
        / f"{filename}.png"
    )

    pdf_path = (
        output_directory
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

    plt.close(fig)

    print(
        f"[DONE] {png_path.name}"
    )

    print(
        f"[DONE] {pdf_path.name}"
    )


# ============================================================
# COMMON AXIS
# ============================================================

def configure_axis(
    ax,
    title,
    ylabel
):

    ax.set_title(
        title,
        fontsize=15,
        fontweight="bold",
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
        0,
        100
    )

    ax.grid(
        True,
        alpha=0.22
    )

    # --------------------------------------------------------
    # Squat bottom.
    # --------------------------------------------------------

    ax.axvline(
        BOTTOM_PERCENT,
        linestyle="--",
        linewidth=1.0,
        alpha=0.65
    )

    ax.text(
        BOTTOM_PERCENT,
        0.97,
        "Approx. squat bottom",
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=9
    )


# ============================================================
# SIDE VIEW
# SINGLE JOINT TRAJECTORY
# ============================================================

def plot_side_joint(
    df,
    column,
    title,
    filename
):

    stats = get_statistics(
        df,
        column
    )

    if stats is None:

        return

    fig, ax = plt.subplots(
        figsize=FIGSIZE
    )

    # --------------------------------------------------------
    # Individual repetition curves.
    # --------------------------------------------------------

    repetitions = (
        df[
            "unique_rep"
        ]
        .drop_duplicates()
        .tolist()
    )

    for rep in repetitions:

        rep_df = df[
            df["unique_rep"]
            == rep
        ].copy()

        rep_df[column] = pd.to_numeric(
            rep_df[column],
            errors="coerce"
        )

        rep_df = rep_df.dropna(
            subset=[
                "cycle_percent",
                column,
            ]
        )

        ax.plot(
            rep_df["cycle_percent"],
            rep_df[column],
            linewidth=0.6,
            alpha=0.10
        )

    # --------------------------------------------------------
    # Mean trajectory.
    # --------------------------------------------------------

    ax.plot(
        stats["cycle"],
        stats["mean"],
        linewidth=3.0,
        label="Mean"
    )

    # --------------------------------------------------------
    # Standard deviation.
    # --------------------------------------------------------

    ax.fill_between(
        stats["cycle"],
        stats["lower"],
        stats["upper"],
        alpha=0.18,
        label="±1 SD"
    )

    configure_axis(
        ax,
        title,
        "Angle (degrees)"
    )

    ax.legend(
        frameon=False
    )

    fig.tight_layout()

    save_figure(
        fig,
        ANGLE_GRAPHS,
        filename
    )


# ============================================================
# FRONT VIEW
# LEFT VS RIGHT
# ============================================================

def plot_front_left_right(
    df,
    left_column,
    right_column,
    joint_name,
    filename
):

    left_stats = get_statistics(
        df,
        left_column
    )

    right_stats = get_statistics(
        df,
        right_column
    )

    if left_stats is None:

        return

    if right_stats is None:

        return

    fig, ax = plt.subplots(
        figsize=FIGSIZE
    )

    # --------------------------------------------------------
    # LEFT
    # --------------------------------------------------------

    ax.plot(
        left_stats["cycle"],
        left_stats["mean"],
        linewidth=3.0,
        label="Left"
    )

    ax.fill_between(
        left_stats["cycle"],
        left_stats["lower"],
        left_stats["upper"],
        alpha=0.12
    )

    # --------------------------------------------------------
    # RIGHT
    # --------------------------------------------------------

    ax.plot(
        right_stats["cycle"],
        right_stats["mean"],
        linewidth=3.0,
        linestyle="--",
        label="Right"
    )

    ax.fill_between(
        right_stats["cycle"],
        right_stats["lower"],
        right_stats["upper"],
        alpha=0.12
    )

    configure_axis(
        ax,
        f"Front View — {joint_name} Left vs Right",
        "Projected angle (degrees)"
    )

    ax.legend(
        frameon=False
    )

    fig.tight_layout()

    save_figure(
        fig,
        ANGLE_GRAPHS,
        filename
    )


# ============================================================
# FRONT VIEW
# ASYMMETRY
# ============================================================

def plot_asymmetry(
    df,
    left_column,
    right_column,
    joint_name,
    filename
):

    if not check_column(
        df,
        left_column
    ):

        return

    if not check_column(
        df,
        right_column
    ):

        return

    temp = df[
        [
            "unique_rep",
            "cycle_percent",
            left_column,
            right_column,
        ]
    ].copy()

    temp[left_column] = pd.to_numeric(
        temp[left_column],
        errors="coerce"
    )

    temp[right_column] = pd.to_numeric(
        temp[right_column],
        errors="coerce"
    )

    # --------------------------------------------------------
    # Absolute left-right difference.
    #
    # This is a descriptive asymmetry measure.
    # It is NOT a clinical injury threshold.
    # --------------------------------------------------------

    temp["asymmetry"] = (
        temp[left_column]
        -
        temp[right_column]
    ).abs()

    temp = temp.dropna(
        subset=[
            "cycle_percent",
            "asymmetry",
        ]
    )

    if temp.empty:

        print(
            f"[WARNING] No asymmetry data for {joint_name}"
        )

        return

    grouped = (
        temp
        .groupby(
            "cycle_percent"
        )["asymmetry"]
    )

    stats = pd.DataFrame({

        "cycle":
            grouped.mean().index,

        "mean":
            grouped.mean().values,

        "std":
            grouped.std(
                ddof=1
            ).fillna(0).values,
    })

    stats["upper"] = (
        stats["mean"]
        +
        stats["std"]
    )

    stats["lower"] = np.maximum(
        stats["mean"]
        -
        stats["std"],
        0
    )

    fig, ax = plt.subplots(
        figsize=FIGSIZE
    )

    # --------------------------------------------------------
    # Individual asymmetry curves.
    # --------------------------------------------------------

    repetitions = (
        temp[
            "unique_rep"
        ]
        .drop_duplicates()
        .tolist()
    )

    for rep in repetitions:

        rep_df = temp[
            temp["unique_rep"]
            == rep
        ]

        ax.plot(
            rep_df["cycle_percent"],
            rep_df["asymmetry"],
            linewidth=0.6,
            alpha=0.10
        )

    # --------------------------------------------------------
    # Mean asymmetry.
    # --------------------------------------------------------

    ax.plot(
        stats["cycle"],
        stats["mean"],
        linewidth=3.0,
        label="Mean absolute difference"
    )

    ax.fill_between(
        stats["cycle"],
        stats["lower"],
        stats["upper"],
        alpha=0.18,
        label="±1 SD"
    )

    configure_axis(
        ax,
        f"Front View — {joint_name} Asymmetry",
        "Absolute left-right angle difference (degrees)"
    )

    ax.legend(
        frameon=False
    )

    fig.tight_layout()

    save_figure(
        fig,
        ANGLE_GRAPHS,
        filename
    )


# ============================================================
# SIDE VIEW
# LOWER LIMB OVERVIEW
# ============================================================

def plot_side_lower_limb(
    df
):

    columns = [
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
        figsize=FIGSIZE
    )

    plotted = 0

    for column, label in columns:

        stats = get_statistics(
            df,
            column
        )

        if stats is None:

            continue

        ax.plot(
            stats["cycle"],
            stats["mean"],
            linewidth=2.5,
            label=label
        )

        plotted += 1

    if plotted == 0:

        plt.close(fig)

        return

    configure_axis(
        ax,
        "Side View — Lower-Limb Joint Kinematics",
        "Angle (degrees)"
    )

    ax.legend(
        frameon=False
    )

    fig.tight_layout()

    save_figure(
        fig,
        COMPARISON_GRAPHS,
        "side_lower_limb_trajectories"
    )


# ============================================================
# FRONT VIEW
# ASYMMETRY OVERVIEW
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
        figsize=FIGSIZE
    )

    plotted = 0

    for left, right, label in joints:

        if not check_column(
            df,
            left
        ):

            continue

        if not check_column(
            df,
            right
        ):

            continue

        temp = df[
            [
                "cycle_percent",
                left,
                right,
            ]
        ].copy()

        temp[left] = pd.to_numeric(
            temp[left],
            errors="coerce"
        )

        temp[right] = pd.to_numeric(
            temp[right],
            errors="coerce"
        )

        temp["asymmetry"] = (
            temp[left]
            -
            temp[right]
        ).abs()

        temp = temp.dropna()

        if temp.empty:

            continue

        mean_curve = (
            temp
            .groupby(
                "cycle_percent"
            )["asymmetry"]
            .mean()
        )

        ax.plot(
            mean_curve.index,
            mean_curve.values,
            linewidth=2.5,
            label=label
        )

        plotted += 1

    if plotted == 0:

        plt.close(fig)

        return

    configure_axis(
        ax,
        "Front View — Joint Asymmetry Across Squat Cycle",
        "Absolute left-right angle difference (degrees)"
    )

    ax.legend(
        frameon=False
    )

    fig.tight_layout()

    save_figure(
        fig,
        COMPARISON_GRAPHS,
        "front_asymmetry_overview"
    )


# ============================================================
# DATASET SUMMARY
# ============================================================

def print_summary(
    front,
    side
):

    print()
    print("=" * 70)
    print("DATASET SUMMARY")
    print("=" * 70)

    for name, df in [
        ("FRONT", front),
        ("SIDE", side),
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

        rows = len(df)

        print()
        print(
            f"{name} VIEW"
        )

        print(
            f"Participants : {participants}"
        )

        print(
            f"Repetitions  : {repetitions}"
        )

        print(
            f"Rows         : {rows}"
        )

        print(
            f"Points/rep   : {rows // repetitions}"
        )


# ============================================================
# COLUMN REPORT
# ============================================================

def print_column_report(
    front,
    side
):

    print()
    print("=" * 70)
    print("AVAILABLE ANGLE COLUMNS")
    print("=" * 70)

    print()
    print("FRONT:")

    for column in front.columns:

        if "angle" in column.lower():

            print(
                f"  {column}"
            )

    print()
    print("SIDE:")

    for column in side.columns:

        if "angle" in column.lower():

            print(
                f"  {column}"
            )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print(
        "SQUAT DATASET — PUBLICATION FIGURES"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    front = load_dataset(
        "front"
    )

    side = load_dataset(
        "side"
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print_summary(
        front,
        side
    )

    print_column_report(
        front,
        side
    )

    # ========================================================
    # SIDE VIEW
    # ========================================================

    print()
    print("=" * 70)
    print(
        "SIDE VIEW — SAGITTAL KINEMATICS"
    )
    print("=" * 70)

    plot_side_joint(
        side,
        "left_knee_angle_deg",
        "Side View — Knee Angle Trajectory",
        "side_knee_trajectory"
    )

    plot_side_joint(
        side,
        "left_hip_angle_deg",
        "Side View — Hip Angle Trajectory",
        "side_hip_trajectory"
    )

    plot_side_joint(
        side,
        "left_ankle_angle_deg",
        "Side View — Ankle Angle Trajectory",
        "side_ankle_trajectory"
    )

    plot_side_joint(
        side,
        "trunk_angle_deg",
        "Side View — Trunk Angle Trajectory",
        "side_trunk_trajectory"
    )

    # ========================================================
    # FRONT VIEW — LEFT VS RIGHT
    # ========================================================

    print()
    print("=" * 70)
    print(
        "FRONT VIEW — LEFT VS RIGHT"
    )
    print("=" * 70)

    plot_front_left_right(
        front,
        "left_knee_angle_deg",
        "right_knee_angle_deg",
        "Knee",
        "front_knee_left_right"
    )

    plot_front_left_right(
        front,
        "left_hip_angle_deg",
        "right_hip_angle_deg",
        "Hip",
        "front_hip_left_right"
    )

    plot_front_left_right(
        front,
        "left_ankle_angle_deg",
        "right_ankle_angle_deg",
        "Ankle",
        "front_ankle_left_right"
    )

    plot_side_joint(
        front,
        "trunk_angle_deg",
        "Front View — Trunk Angle Trajectory",
        "front_trunk_trajectory"
    )

    # ========================================================
    # FRONT VIEW — ASYMMETRY
    # ========================================================

    print()
    print("=" * 70)
    print(
        "FRONT VIEW — LEFT-RIGHT ASYMMETRY"
    )
    print("=" * 70)

    plot_asymmetry(
        front,
        "left_knee_angle_deg",
        "right_knee_angle_deg",
        "Knee",
        "front_knee_asymmetry"
    )

    plot_asymmetry(
        front,
        "left_hip_angle_deg",
        "right_hip_angle_deg",
        "Hip",
        "front_hip_asymmetry"
    )

    plot_asymmetry(
        front,
        "left_ankle_angle_deg",
        "right_ankle_angle_deg",
        "Ankle",
        "front_ankle_asymmetry"
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
        side
    )

    plot_front_asymmetry_overview(
        front
    )

    # ========================================================
    # FINAL
    # ========================================================

    print()
    print("=" * 70)
    print(
        "FIGURE GENERATION COMPLETE"
    )
    print("=" * 70)

    print()
    print(
        "ANGLE GRAPHS:"
    )

    print(
        ANGLE_GRAPHS
    )

    print()
    print(
        "COMPARISON GRAPHS:"
    )

    print(
        COMPARISON_GRAPHS
    )

    print()
    print(
        "Interpretation:"
    )

    print(
        "Side view -> sagittal-plane squat kinematics"
    )

    print(
        "Front view -> projected left-right movement and asymmetry"
    )

    print(
        "No clinical injury diagnosis is assigned."
    )

    print()


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()