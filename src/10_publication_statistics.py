from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# SQUAT DATASET — PUBLICATION STATISTICS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

FEATURE_ROOT = ROOT / "data" / "processed" / "features"
RISK_ROOT = ROOT / "data" / "processed" / "risk"

OUTPUT_ROOT = ROOT / "data" / "processed" / "publication"

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# HELPER
# ============================================================

def numeric(series):
    return pd.to_numeric(
        series,
        errors="coerce"
    )


# ============================================================
# BASIC DATASET SUMMARY
# ============================================================

def create_dataset_summary():

    rows = []

    for view in ["front", "side"]:

        path = (
            FEATURE_ROOT
            / f"{view}_features.csv"
        )

        df = pd.read_csv(path)

        participants = (
            df["participant"]
            .nunique()
        )

        repetitions = len(df)

        rows.append({
            "view": view,
            "participants": participants,
            "repetitions": repetitions,
            "features_per_repetition": len(df.columns),
        })

    summary = pd.DataFrame(rows)

    path = (
        OUTPUT_ROOT
        / "dataset_summary.csv"
    )

    summary.to_csv(
        path,
        index=False
    )

    print(
        f"[DONE] {path}"
    )


# ============================================================
# JOINT FEATURE SUMMARY
# ============================================================

def create_joint_summary():

    df = pd.read_csv(
        FEATURE_ROOT
        / "front_features.csv"
    )

    joints = [
        ("Knee", "left_knee", "right_knee"),
        ("Hip", "left_hip", "right_hip"),
        ("Ankle", "left_ankle", "right_ankle"),
    ]

    rows = []

    for joint, left, right in joints:

        left_values = numeric(
            df[f"{left}_mean_deg"]
        )

        right_values = numeric(
            df[f"{right}_mean_deg"]
        )

        left_rom = numeric(
            df[f"{left}_range_deg"]
        )

        right_rom = numeric(
            df[f"{right}_range_deg"]
        )

        rows.append({

            "joint": joint,

            "left_mean_angle_deg":
                left_values.mean(),

            "right_mean_angle_deg":
                right_values.mean(),

            "left_sd_deg":
                left_values.std(),

            "right_sd_deg":
                right_values.std(),

            "left_mean_rom_deg":
                left_rom.mean(),

            "right_mean_rom_deg":
                right_rom.mean(),

            "left_bottom_angle_deg":
                numeric(
                    df[
                        f"{left}_bottom_angle_deg"
                    ]
                ).mean(),

            "right_bottom_angle_deg":
                numeric(
                    df[
                        f"{right}_bottom_angle_deg"
                    ]
                ).mean(),
        })

    summary = pd.DataFrame(rows)

    path = (
        OUTPUT_ROOT
        / "joint_kinematic_summary.csv"
    )

    summary.to_csv(
        path,
        index=False
    )

    print(
        f"[DONE] {path}"
    )


# ============================================================
# FRONT ASYMMETRY SUMMARY
# ============================================================

def create_front_asymmetry_summary():

    path = (
        RISK_ROOT
        / "front_risk_indicators.csv"
    )

    df = pd.read_csv(path)

    indicators = [
        (
            "Knee",
            "knee_asymmetry_deg"
        ),
        (
            "Hip",
            "hip_asymmetry_deg"
        ),
        (
            "Ankle",
            "ankle_asymmetry_deg"
        ),
    ]

    rows = []

    for joint, column in indicators:

        values = numeric(
            df[column]
        ).dropna()

        threshold = np.percentile(
            values,
            95
        )

        upper_count = int(
            (
                values >= threshold
            ).sum()
        )

        rows.append({

            "joint": joint,

            "mean_asymmetry_deg":
                values.mean(),

            "median_asymmetry_deg":
                values.median(),

            "std_deg":
                values.std(),

            "minimum_deg":
                values.min(),

            "maximum_deg":
                values.max(),

            "95th_percentile_deg":
                threshold,

            "upper_5_percent_count":
                upper_count,

            "upper_5_percent_percentage":
                upper_count
                /
                len(values)
                *
                100,
        })

    summary = pd.DataFrame(rows)

    path = (
        OUTPUT_ROOT
        / "front_asymmetry_summary.csv"
    )

    summary.to_csv(
        path,
        index=False
    )

    print(
        f"[DONE] {path}"
    )


# ============================================================
# SIDE TRUNK SUMMARY
# ============================================================

def create_trunk_summary():

    path = (
        RISK_ROOT
        / "side_risk_indicators.csv"
    )

    df = pd.read_csv(path)

    values = numeric(
        df[
            "trunk_excursion_deg"
        ]
    ).dropna()

    threshold = np.percentile(
        values,
        95
    )

    upper_count = int(
        (
            values >= threshold
        ).sum()
    )

    summary = pd.DataFrame([{

        "mean_trunk_excursion_deg":
            values.mean(),

        "median_trunk_excursion_deg":
            values.median(),

        "std_deg":
            values.std(),

        "minimum_deg":
            values.min(),

        "maximum_deg":
            values.max(),

        "95th_percentile_deg":
            threshold,

        "upper_5_percent_count":
            upper_count,

        "upper_5_percent_percentage":
            upper_count
            /
            len(values)
            *
            100,
    }])

    path = (
        OUTPUT_ROOT
        / "trunk_movement_summary.csv"
    )

    summary.to_csv(
        path,
        index=False
    )

    print(
        f"[DONE] {path}"
    )


# ============================================================
# FEATURE STATISTICS
# ============================================================

def create_feature_statistics():

    df = pd.read_csv(
        FEATURE_ROOT
        / "front_features.csv"
    )

    numeric_columns = (
        df.select_dtypes(
            include=np.number
        )
        .columns
    )

    rows = []

    for column in numeric_columns:

        values = numeric(
            df[column]
        ).dropna()

        if values.empty:
            continue

        rows.append({

            "feature": column,

            "count":
                len(values),

            "mean":
                values.mean(),

            "std":
                values.std(),

            "minimum":
                values.min(),

            "25_percentile":
                values.quantile(0.25),

            "median":
                values.median(),

            "75_percentile":
                values.quantile(0.75),

            "maximum":
                values.max(),
        })

    summary = pd.DataFrame(rows)

    path = (
        OUTPUT_ROOT
        / "feature_statistics.csv"
    )

    summary.to_csv(
        path,
        index=False
    )

    print(
        f"[DONE] {path}"
    )


# ============================================================
# PARTICIPANT-LEVEL SUMMARY
# ============================================================

def create_participant_summary():

    df = pd.read_csv(
        FEATURE_ROOT
        / "front_features.csv"
    )

    summary = (
        df.groupby(
            "participant"
        )
        .agg(
            repetitions=(
                "rep_id",
                "count"
            )
        )
        .reset_index()
    )

    path = (
        OUTPUT_ROOT
        / "participant_summary.csv"
    )

    summary.to_csv(
        path,
        index=False
    )

    print(
        f"[DONE] {path}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print(
        "SQUAT DATASET — PUBLICATION STATISTICS"
    )
    print("=" * 70)

    print()
    print(
        "Generating dataset summary..."
    )

    create_dataset_summary()

    print()
    print(
        "Generating joint kinematic summary..."
    )

    create_joint_summary()

    print()
    print(
        "Generating front-view asymmetry summary..."
    )

    create_front_asymmetry_summary()

    print()
    print(
        "Generating side-view trunk summary..."
    )

    create_trunk_summary()

    print()
    print(
        "Generating feature statistics..."
    )

    create_feature_statistics()

    print()
    print(
        "Generating participant summary..."
    )

    create_participant_summary()

    print()
    print("=" * 70)
    print(
        "PUBLICATION STATISTICS COMPLETE"
    )
    print("=" * 70)

    print()
    print(
        "Output directory:"
    )

    print(
        OUTPUT_ROOT
    )

    print()
    print(
        "Generated tables:"
    )

    for file in sorted(
        OUTPUT_ROOT.glob("*.csv")
    ):

        print(
            f"  {file.name}"
        )

    print()


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()