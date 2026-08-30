from pathlib import Path
import pandas as pd
import numpy as np


ROOT = Path(__file__).resolve().parents[1]

INPUT_ROOT = ROOT / "data" / "processed" / "normalized"
OUTPUT_ROOT = ROOT / "data" / "processed" / "features"

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


def extract_features(view):

    input_file = (
        INPUT_ROOT
        / view
        / "all_normalized_repetitions.csv"
    )

    output_file = (
        OUTPUT_ROOT
        / f"{view}_features.csv"
    )

    print()
    print("=" * 70)
    print(f"FEATURE EXTRACTION — {view.upper()} VIEW")
    print("=" * 70)

    if not input_file.exists():
        print(f"[ERROR] File not found:")
        print(input_file)
        return

    df = pd.read_csv(input_file)

    # Create unique repetition identifier
    if "participant" in df.columns and "rep_id" in df.columns:
        df["unique_rep"] = (
            df["participant"].astype(str)
            + "_"
            + df["rep_id"].astype(str)
        )

    angle_columns = [
        c for c in df.columns
        if c.endswith("_angle_deg")
    ]

    if not angle_columns:
        print("[ERROR] No angle columns found.")
        print(df.columns.tolist())
        return

    print(f"Rows: {len(df)}")
    print(f"Angle columns: {angle_columns}")

    features = []

    for rep_id, rep in df.groupby("unique_rep"):

        row = {
            "unique_rep": rep_id
        }

        # Participant
        if "participant" in rep.columns:
            row["participant"] = rep["participant"].iloc[0]

        # Repetition
        if "rep_id" in rep.columns:
            row["rep_id"] = rep["rep_id"].iloc[0]

        # --------------------------------------------------
        # Extract features for every angle
        # --------------------------------------------------

        for column in angle_columns:

            values = pd.to_numeric(
                rep[column],
                errors="coerce"
            ).dropna()

            if values.empty:
                continue

            joint_name = column.replace(
                "_angle_deg",
                ""
            )

            row[f"{joint_name}_min_deg"] = values.min()

            row[f"{joint_name}_max_deg"] = values.max()

            row[f"{joint_name}_mean_deg"] = values.mean()

            row[f"{joint_name}_std_deg"] = values.std()

            row[f"{joint_name}_range_deg"] = (
                values.max() - values.min()
            )

            # Value closest to normalized squat bottom
            if "cycle_percent" in rep.columns:

                temp = rep[
                    [
                        "cycle_percent",
                        column
                    ]
                ].copy()

                temp[column] = pd.to_numeric(
                    temp[column],
                    errors="coerce"
                )

                temp = temp.dropna()

                if not temp.empty:

                    bottom_index = (
                        temp["cycle_percent"]
                        - 50
                    ).abs().idxmin()

                    row[
                        f"{joint_name}_bottom_angle_deg"
                    ] = temp.loc[
                        bottom_index,
                        column
                    ]

        features.append(row)

    feature_df = pd.DataFrame(features)

    feature_df.to_csv(
        output_file,
        index=False
    )

    print()
    print(f"[DONE] Saved:")
    print(output_file)

    print()
    print(f"Repetitions: {len(feature_df)}")
    print(f"Features: {len(feature_df.columns)}")


def main():

    print()
    print("=" * 70)
    print("SQUAT DATASET — FEATURE EXTRACTION")
    print("=" * 70)

    extract_features("front")

    extract_features("side")

    print()
    print("=" * 70)
    print("FEATURE EXTRACTION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()