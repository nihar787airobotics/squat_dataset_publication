"""
04_segment_squats.py

Squat Dataset Publication Pipeline
----------------------------------

Purpose:
    Detect individual squat repetitions from the calculated
    knee-angle trajectories.

Method:
    1. Load joint-angle CSV.
    2. Select the most reliable knee signal.
    3. Smooth the knee-angle trajectory.
    4. Detect local minima corresponding to squat bottoms.
    5. Determine the beginning and end of each repetition.
    6. Save repetition-level segmentation metadata.
    7. Save frame-level repetition labels.

Important:
    No fixed universal knee-angle threshold is used.
    The detection is adaptive to each recording.

Input:
    data/processed/angles/front/*.csv
    data/processed/angles/side/*.csv

Output:
    data/processed/segments/front/*.csv
    data/processed/segments/side/*.csv

Each output CSV contains:
    frame
    timestamp_ms
    rep_id
    phase
    knee_angle_deg
    squat_depth_angle_deg
"""

from pathlib import Path

import numpy as np
import pandas as pd

from scipy.signal import (
    find_peaks,
    savgol_filter
)

from tqdm import tqdm


# ============================================================
# 1. PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

INPUT_DIR = (
    ROOT
    / "data"
    / "processed"
    / "angles"
)

OUTPUT_DIR = (
    ROOT
    / "data"
    / "processed"
    / "segments"
)


# ============================================================
# 2. PARAMETERS
# ============================================================

# Minimum meaningful knee-angle excursion.
#
# A squat must show at least this much reduction from
# the estimated standing knee angle.
MIN_EXCURSION_DEG = 15.0

# Minimum separation between two squat bottoms.
#
# This is deliberately conservative to avoid counting
# tiny oscillations as separate repetitions.
MIN_REP_DISTANCE_SECONDS = 0.7

# Fraction of the standing-to-bottom excursion used to
# define the beginning/end of a squat.
#
# 0.25 means the signal must move approximately 25%
# from standing toward the bottom before the repetition
# is considered active.
BOUNDARY_FRACTION = 0.25

# Savitzky-Golay smoothing window.
SMOOTHING_WINDOW = 11

# Polynomial order for smoothing.
SMOOTHING_POLYORDER = 2


# ============================================================
# 3. UTILITY FUNCTIONS
# ============================================================

def safe_savgol(signal):
    """
    Smooth a signal while automatically adapting the window
    to the available number of samples.
    """

    signal = np.asarray(
        signal,
        dtype=float
    )

    n = len(signal)

    if n < 7:
        return signal.copy()

    window = min(
        SMOOTHING_WINDOW,
        n if n % 2 == 1 else n - 1
    )

    if window < 5:
        return signal.copy()

    polyorder = min(
        SMOOTHING_POLYORDER,
        window - 1
    )

    return savgol_filter(
        signal,
        window_length=window,
        polyorder=polyorder
    )


# ============================================================
# 4. SELECT KNEE SIGNAL
# ============================================================

def create_knee_signal(df):
    """
    Create the most reliable knee-angle signal.

    If both knees are available, use their mean.

    If only one knee is available, use that knee.

    This is important because a single camera view can
    occasionally produce missing landmark measurements.
    """

    left = pd.to_numeric(
        df["left_knee_angle_deg"],
        errors="coerce"
    )

    right = pd.to_numeric(
        df["right_knee_angle_deg"],
        errors="coerce"
    )

    knee = pd.concat(
        [left, right],
        axis=1
    ).mean(
        axis=1,
        skipna=True
    )

    return knee


# ============================================================
# 5. ESTIMATE STANDING BASELINE
# ============================================================

def estimate_standing_angle(knee_signal):
    """
    Estimate the standing knee angle.

    During standing, the knee angle is generally among the
    larger values in the recording.

    The 90th percentile provides a robust estimate without
    requiring a manually selected standing interval.
    """

    valid = knee_signal[
        np.isfinite(knee_signal)
    ]

    if len(valid) == 0:
        return np.nan

    return float(
        np.percentile(
            valid,
            90
        )
    )


# ============================================================
# 6. DETECT SQUAT BOTTOMS
# ============================================================

def detect_bottoms(
    knee_signal,
    fps
):
    """
    Detect local minima of knee angle.

    A squat bottom corresponds approximately to maximum
    knee flexion and therefore a local minimum in knee angle.
    """

    valid_mask = np.isfinite(
        knee_signal
    )

    if valid_mask.sum() < 10:
        return []

    # Fill short missing regions by interpolation.
    signal = (
        pd.Series(knee_signal)
        .interpolate(
            limit_direction="both"
        )
        .to_numpy()
    )

    smoothed = safe_savgol(
        signal
    )

    standing_angle = (
        estimate_standing_angle(
            smoothed
        )
    )

    if np.isnan(standing_angle):
        return []

    # Determine the largest meaningful excursion.
    minimum_angle = float(
        np.min(smoothed)
    )

    max_excursion = (
        standing_angle
        - minimum_angle
    )

    # Reject recordings without a meaningful squat.
    if max_excursion < MIN_EXCURSION_DEG:
        return []

    # Adaptive prominence.
    prominence = max(
        MIN_EXCURSION_DEG * 0.35,
        max_excursion * 0.15
    )

    min_distance = max(
        1,
        int(
            fps
            * MIN_REP_DISTANCE_SECONDS
        )
    )

    peaks, properties = find_peaks(
        -smoothed,
        distance=min_distance,
        prominence=prominence
    )

    # Keep only minima showing sufficient flexion.
    valid_bottoms = []

    for peak in peaks:

        angle = smoothed[peak]

        excursion = (
            standing_angle
            - angle
        )

        if excursion >= MIN_EXCURSION_DEG:

            valid_bottoms.append(
                int(peak)
            )

    return valid_bottoms


# ============================================================
# 7. FIND REPETITION BOUNDARIES
# ============================================================

def find_boundaries(
    signal,
    bottom_index,
    standing_angle
):
    """
    Find the start and end of a repetition around a squat
    bottom.

    Boundary level:

        standing angle
             |
             |\
             | \
             |  \  descent
             |   \
             |    \____ bottom
             |         \
             |          \
             |           \ ascent
             |
             +----------------

    The boundary is determined adaptively from the person's
    own standing-to-bottom excursion.
    """

    n = len(signal)

    bottom_angle = signal[
        bottom_index
    ]

    excursion = (
        standing_angle
        - bottom_angle
    )

    if excursion <= 0:
        return None, None

    boundary_angle = (
        bottom_angle
        +
        BOUNDARY_FRACTION
        * excursion
    )

    # --------------------------------------------------------
    # Search backwards for descent boundary.
    # --------------------------------------------------------

    start = 0

    for i in range(
        bottom_index - 1,
        -1,
        -1
    ):

        if signal[i] >= boundary_angle:

            start = i

            break

    # --------------------------------------------------------
    # Search forwards for ascent boundary.
    # --------------------------------------------------------

    end = n - 1

    for i in range(
        bottom_index + 1,
        n
    ):

        if signal[i] >= boundary_angle:

            end = i

            break

    return start, end


# ============================================================
# 8. CREATE REPETITION TABLE
# ============================================================

def create_repetition_table(
    df,
    knee_signal,
    bottoms,
    fps
):
    """
    Create one row per detected squat repetition.
    """

    if not bottoms:
        return pd.DataFrame()

    smoothed = safe_savgol(
        knee_signal
    )

    standing_angle = (
        estimate_standing_angle(
            smoothed
        )
    )

    repetitions = []

    for rep_number, bottom in enumerate(
        bottoms,
        start=1
    ):

        start, end = find_boundaries(
            smoothed,
            bottom,
            standing_angle
        )

        if start is None or end is None:
            continue

        if end <= start:
            continue

        bottom_angle = float(
            smoothed[bottom]
        )

        excursion = (
            standing_angle
            - bottom_angle
        )

        start_frame = int(
            df.iloc[start]["frame"]
        )

        bottom_frame = int(
            df.iloc[bottom]["frame"]
        )

        end_frame = int(
            df.iloc[end]["frame"]
        )

        start_time = float(
            df.iloc[start]["timestamp_ms"]
        )

        bottom_time = float(
            df.iloc[bottom]["timestamp_ms"]
        )

        end_time = float(
            df.iloc[end]["timestamp_ms"]
        )

        repetitions.append({

            "rep_id":
                rep_number,

            "start_frame":
                start_frame,

            "bottom_frame":
                bottom_frame,

            "end_frame":
                end_frame,

            "start_timestamp_ms":
                start_time,

            "bottom_timestamp_ms":
                bottom_time,

            "end_timestamp_ms":
                end_time,

            "duration_seconds":
                (
                    end_time
                    -
                    start_time
                ) / 1000.0,

            "standing_knee_angle_deg":
                standing_angle,

            "bottom_knee_angle_deg":
                bottom_angle,

            "knee_flexion_excursion_deg":
                excursion,

            "minimum_knee_angle_deg":
                bottom_angle,
        })

    return pd.DataFrame(
        repetitions
    )


# ============================================================
# 9. CREATE FRAME-LEVEL LABELS
# ============================================================

def create_frame_labels(
    df,
    knee_signal,
    repetitions
):
    """
    Assign every frame to a repetition and movement phase.

    Phases:
        outside
        descent
        bottom
        ascent
    """

    output = df[
        [
            "video",
            "view",
            "frame",
            "timestamp_ms",
            "fps",
        ]
    ].copy()

    output[
        "knee_angle_deg"
    ] = knee_signal

    output[
        "rep_id"
    ] = 0

    output[
        "phase"
    ] = "outside"

    for _, rep in repetitions.iterrows():

        start_frame = int(
            rep["start_frame"]
        )

        bottom_frame = int(
            rep["bottom_frame"]
        )

        end_frame = int(
            rep["end_frame"]
        )

        rep_id = int(
            rep["rep_id"]
        )

        mask = (
            (output["frame"] >= start_frame)
            &
            (output["frame"] <= end_frame)
        )

        output.loc[
            mask,
            "rep_id"
        ] = rep_id

        descent_mask = (
            mask
            &
            (
                output["frame"]
                <
                bottom_frame
            )
        )

        output.loc[
            descent_mask,
            "phase"
        ] = "descent"

        bottom_mask = (
            mask
            &
            (
                output["frame"]
                ==
                bottom_frame
            )
        )

        output.loc[
            bottom_mask,
            "phase"
        ] = "bottom"

        ascent_mask = (
            mask
            &
            (
                output["frame"]
                >
                bottom_frame
            )
        )

        output.loc[
            ascent_mask,
            "phase"
        ] = "ascent"

    return output


# ============================================================
# 10. PROCESS ONE FILE
# ============================================================

def process_file(
    input_file,
    view
):
    """
    Process one angle CSV.
    """

    df = pd.read_csv(
        input_file
    )

    if df.empty:

        print(
            f"[WARNING] Empty file: "
            f"{input_file.name}"
        )

        return

    # --------------------------------------------------------
    # Create knee signal
    # --------------------------------------------------------

    knee_signal = create_knee_signal(
        df
    )

    # --------------------------------------------------------
    # FPS
    # --------------------------------------------------------

    fps_values = pd.to_numeric(
        df["fps"],
        errors="coerce"
    ).dropna()

    if len(fps_values) == 0:

        fps = 30.0

    else:

        fps = float(
            fps_values.iloc[0]
        )

    # --------------------------------------------------------
    # Detect bottoms
    # --------------------------------------------------------

    bottoms = detect_bottoms(
        knee_signal.to_numpy(),
        fps
    )

    # --------------------------------------------------------
    # Create repetition summary
    # --------------------------------------------------------

    repetitions = create_repetition_table(
        df,
        knee_signal.to_numpy(),
        bottoms,
        fps
    )

    # --------------------------------------------------------
    # Create frame labels
    # --------------------------------------------------------

    frame_labels = create_frame_labels(
        df,
        knee_signal,
        repetitions
    )

    # --------------------------------------------------------
    # Output directories
    # --------------------------------------------------------

    output_dir = (
        OUTPUT_DIR / view
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Save frame-level segmentation
    # --------------------------------------------------------

    frame_output = (
        output_dir
        / f"{input_file.stem}_frames.csv"
    )

    frame_labels.to_csv(
        frame_output,
        index=False
    )

    # --------------------------------------------------------
    # Save repetition-level metadata
    # --------------------------------------------------------

    repetition_output = (
        output_dir
        / f"{input_file.stem}_repetitions.csv"
    )

    repetitions.to_csv(
        repetition_output,
        index=False
    )

    # --------------------------------------------------------
    # Console information
    # --------------------------------------------------------

    print()

    print(
        f"Video: {input_file.name}"
    )

    print(
        f"Detected repetitions: "
        f"{len(repetitions)}"
    )

    print(
        f"Frame labels saved: "
        f"{frame_output}"
    )

    print(
        f"Repetition metadata saved: "
        f"{repetition_output}"
    )


# ============================================================
# 11. PROCESS VIEW
# ============================================================

def process_view(view):

    input_dir = (
        INPUT_DIR / view
    )

    if not input_dir.exists():

        print(
            f"[WARNING] Missing folder: "
            f"{input_dir}"
        )

        return

    files = sorted(
        input_dir.glob(
            "*.csv"
        )
    )

    print()
    print("=" * 70)

    print(
        f"{view.upper()} VIEW"
    )

    print(
        f"Angle files: {len(files)}"
    )

    print("=" * 70)

    for file in tqdm(
        files,
        desc=f"{view} squat segmentation"
    ):

        process_file(
            file,
            view
        )


# ============================================================
# 12. MAIN
# ============================================================

def main():

    print()
    print("=" * 70)

    print(
        "SQUAT DATASET — REPETITION SEGMENTATION"
    )

    print("=" * 70)

    print()
    print(
        "Detection signal:"
    )

    print(
        "Left/right knee-angle trajectory"
    )

    print()
    print(
        "Method:"
    )

    print(
        "Adaptive knee-flexion minima"
    )

    process_view(
        "front"
    )

    process_view(
        "side"
    )

    print()
    print("=" * 70)

    print(
        "SQUAT SEGMENTATION COMPLETE"
    )

    print("=" * 70)

    print()
    print(
        "Output:"
    )

    print(
        OUTPUT_DIR
    )

    print()


# ============================================================
# PROGRAM ENTRY
# ============================================================

if __name__ == "__main__":

    main()