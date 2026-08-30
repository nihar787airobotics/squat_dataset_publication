# Dual-View Squat Movement Dataset

Computer-vision-based dual-view squat kinematics and reproducible movement-analysis pipeline.

---

## Overview

This repository provides a open-research dataset of synchronized dual-view (frontal and sagittal) 2D RGB video recordings of bodyweight squat movements, paired with an automated, step-by-step Python processing pipeline. Data were acquired from 20 general, non-gym-trained participants (15 male, 5 female) performing 5 continuous squat repetitions per session, producing 100 analyzed repetitions per camera view.

The processing software converts raw video input into planar 2D pose trajectories, calculates continuous joint angles, segments individual squat cycles, and resamples each movement to 101 normalized time points (0% to 100% of cycle completion). From these normalized signals, the pipeline computes 45 scalar kinematic features per repetition, as well as descriptive movement indicators including frontal bilateral asymmetries (knee, hip, ankle) and sagittal trunk forward-lean excursion.

Designed for computer vision benchmarking, biomechanical trajectory modeling, and movement feature engineering, this repository emphasizes complete transparency and scientific reproducibility. All raw inputs, intermediate processing records, and final summary statistics are structured for simple inspection and re-analysis.

> **Scientific Scope:** This dataset consists of dual-view 2D RGB video and pose-derived kinematic trajectories. The derived movement indicators are empirical statistical descriptors of movement variation within this cohort; they are **not** clinical injury diagnoses or medical risk evaluations.

---

## Dataset at a Glance

---

## Experimental Setup

*Figure 1: Illustrative dual-camera acquisition setup. Front and side RGB views were recorded simultaneously while participants performed squat repetitions. Camera geometry is schematic and not drawn to scale.*

* **Camera Equipment:** Mobile capture setup using a Samsung Galaxy S23+ (Model: `SM-S916B/DS`).
* **Environment:** Controlled studio space with a uniform chroma-key background.
* **Camera Views:** Frontal view (positioned directly facing participant) and Sagittal view (positioned orthogonal to participant's left/right plane).
* **Systematic Constraints:** Camera-to-participant distance and camera height were not systematically recorded in absolute physical units.

---

## Processing Pipeline

```text
RAW RGB VIDEO (Front & Side Views)
       │
       ▼
POSE LANDMARK EXTRACTION (MediaPipe Pose Landmarker Full — 33 Landmarks)
       │
       ▼
JOINT-ANGLE CALCULATION (Planar 2D Vector Angles for Knee, Hip, Ankle, Trunk)
       │
       ▼
SQUAT REPETITION SEGMENTATION (Automated Extrema Detection & Boundary Extraction)
       │
       ▼
CYCLE NORMALIZATION (Cubic Spline Resampling to 101 Uniform Time Points)
       │
       ▼
FEATURE EXTRACTION (45 Scalar Metrics per Repetition)
       │
       ▼
MOVEMENT INDICATORS (Frontal Left-Right Asymmetry & Sagittal Trunk Excursion)
       │
       ▼
PUBLICATION STATISTICS & FIGURES (Summary Statistics CSVs, PNG Distributions, PDF Reports)

```

---

## Pose Estimation

Landmark extraction relies on Google's MediaPipe Pose Landmarker Full model architecture. The pipeline processes frame-by-frame RGB input to estimate 33 full-body 3D landmark locations along with visibility scores.

| Parameter | Pipeline Configuration Value |
| --- | --- |
| **Model Architecture** | MediaPipe Pose Landmarker Full |
| **Model Weight File** | `models/pose_landmarker_full.task` |
| **Body Landmarks Extracted** | 33 keypoints |
| **Minimum Detection Confidence** (`min_pose_detection_confidence`) | `0.5` |
| **Minimum Presence Confidence** (`min_pose_presence_confidence`) | `0.5` |
| **Minimum Tracking Confidence** (`min_tracking_confidence`) | `0.5` |

> **Technical Note on Confidence Parameters:** The threshold value of `0.5` represents an internal model confidence cutoff (on a scale of 0.0 to 1.0) required to initiate detection, confirm presence, or sustain keypoint tracking across frames. It is a configuration filter and does **not** indicate a 50% model accuracy rating.

---

## Data Acquisition

Video recordings were captured concurrently across two orthogonal camera angles under standard ambient studio lighting.

* **Front-View Stream:** Processed resolution of 848 × 478 pixels at approximately 60 frames per second (FPS).
* **Side-View Stream:** Processed resolution of 848 × 478 pixels at approximately 30–60 FPS.
* **Recording Protocol:** Participants performed 5 consecutive bodyweight squats per session without external load or artificial pacing constraints.
* **Cohort Demographics:** 20 healthy adults (15 male, 5 female) categorized as general/non-gym-trained participants (0 competitive or trained powerlifters/weightlifters).

---

## Data Processing

The processing architecture consists of seven modular automated stages implemented under `src/`:

### 1. Pose Landmark Extraction (`src/01_extract_keypoints.py`)

* **INPUT:** Raw dual-view `.mp4` video files from `data/raw/`.
* **PROCESS:** Executes MediaPipe Pose Landmarker Full frame-by-frame. Retains spatial $(x, y, z)$ image coordinates and landmark visibility scores.
* **OUTPUT:** Raw keypoint coordinate files in `data/processed/keypoints_raw.csv`.

### 2. Joint-Angle Calculation (`src/02_calculate_angles.py`)

* **INPUT:** Raw landmark trajectory coordinate tables.
* **PROCESS:** Computes 2D planar vector angles for target joint segments: left/right knees, left/right hips, left/right ankles, and trunk forward lean relative to vertical.
* **OUTPUT:** Continuous joint angle time-series in `data/processed/joint_angles.csv`.

### 3. Squat Repetition Segmentation (`src/04_segment_squats.py`)

* **INPUT:** Continuous knee joint angle time-series.
* **PROCESS:** Identifies repetition start, bottom inflection point (maximum flexion), and extension completion using localized extrema detection algorithm.
* **OUTPUT:** Repetition boundary indices and discrete repetition segments.

### 4. Cycle Normalization (`src/05_normalize_cycles.py`)

* **INPUT:** Variable-length segmented repetition time-series.
* **PROCESS:** Applies cubic spline interpolation to convert each repetition into exactly 101 standardized percentage points (0% = start, ~50% = deepest bottom phase, 100% = finish).
* **OUTPUT:** Normalized kinematic matrix ($10,100 \text{ rows} = 100 \text{ repetitions} \times 101 \text{ points}$) in `data/processed/squat_cycles_normalized.csv`.

### 5. Feature Extraction (`src/06_extract_features.py`)

* **INPUT:** 101-point time-normalized kinematic curves.
* **PROCESS:** Computes 45 summary descriptors per repetition including joint minimums, maximums, means, standard deviations, range of motion (ROM), and bottom-phase inflection angles.
* **OUTPUT:** Repetition feature matrix in `data/processed/kinematic_features_45.csv`.

### 6. Movement-Indicator Generation (`src/07_risk_patterns.py`)

* **INPUT:** Frontal joint angle curves and sagittal trunk trajectories.
* **PROCESS:** Evaluates bilateral absolute asymmetry $\vert{}\theta_{\text{left}} - \theta_{\text{right}}\vert{}$ for lower-limb joints and trunk excursion (total change in forward lean angle during movement).
* **OUTPUT:** Repetition indicator values in `outputs/movement_indicators.csv`.

### 7. Publication Statistics (`src/10_publication_statistics.py`)

* **INPUT:** Feature tables and movement indicator CSVs.
* **PROCESS:** Calculates aggregate cohort statistics (mean, SD, median, range, 95th percentile) and exports publication-ready summaries and figures.
* **OUTPUT:** Summary metrics in `outputs/publication_summary_stats.csv` and graphical plots in `outputs/figures/`.

---

## Repository Structure

```text
squat_dataset_publication/
├── data/
│   ├── raw/                         # Raw MP4 video files (front and side views)
│   └── processed/                   # Keypoints, continuous angles, normalized cycles, feature CSVs
├── docs/
│   └── images/                      # Setup schematics, pipeline visual diagrams, and architecture graphics
│       ├── experimental_setup.png
│       ├── pipeline.png
│       └── data_hierarchy.png
├── models/
│   └── pose_landmarker_full.task    # Pre-trained MediaPipe Pose Landmarker Full binary model file
├── outputs/
│   ├── figures/                     # Generated visual distribution plots and trajectory figures (.png)
│   ├── movement_indicators.csv      # Derived asymmetry metrics and trunk excursion values
│   └── publication_summary_stats.csv# Aggregated statistical summary tables
├── src/
│   ├── 01_extract_keypoints.py      # Extracts MediaPipe pose landmarks from video
│   ├── 02_calculate_angles.py       # Computes 2D planar joint angle time-series
│   ├── 04_segment_squats.py         # Detects repetition start/inflection/end boundaries
│   ├── 05_normalize_cycles.py       # Resamples cycles to 101 uniform percentage points
│   ├── 06_extract_features.py       # Calculates 45 repetition-level kinematic features
│   ├── 07_risk_patterns.py          # Derives bilateral asymmetries and trunk excursion metrics
│   ├── 08_generate_figures.py       # Generates publication visual figures and distribution plots
│   └── 10_publication_statistics.py # Computes cohort descriptive summary statistics
├── requirements.txt                 # Python runtime dependencies
└── README.md                        # Primary dataset repository documentation

```

---

## Data Records

The table below details the output artifacts produced by the processing pipeline:

| Data Product | Relative Storage Location | Description |
| --- | --- | --- |
| **Raw Video Stream** | `data/raw/*.mp4` | Original synchronized dual-view MP4 video recordings |
| **Raw Keypoints** | `data/processed/keypoints_raw.csv` | Frame-by-frame $(x, y, z)$ coordinates and landmark visibility |
| **Joint Angles** | `data/processed/joint_angles.csv` | Unsegmented continuous planar joint angle signals |
| **Normalized Cycles** | `data/processed/squat_cycles_normalized.csv` | 101-point resampled kinematics ($100 \text{ reps} \times 101 \text{ points}$) |
| **Kinematic Features** | `data/processed/kinematic_features_45.csv` | 45 extracted scalar descriptors per repetition |
| **Movement Indicators** | `outputs/movement_indicators.csv` | Frontal asymmetries and sagittal trunk excursion metrics |
| **Summary Statistics** | `outputs/publication_summary_stats.csv` | Cohort-level summary metrics (mean, SD, median, 95th percentile) |
| **Publication Figures** | `outputs/figures/*.png` | Visual figures, trajectory plots, and distribution graphs |

---

## Feature Dataset

The feature extraction script (`src/06_extract_features.py`) calculates **45 scalar features** per repetition. These features capture kinematic characteristics across the flexion-extension movement:

```text
45 Extracted Features Structure:
├── Left Knee Kinematics  (6) : Min, Max, Mean, SD, ROM, Bottom Angle
├── Right Knee Kinematics (6) : Min, Max, Mean, SD, ROM, Bottom Angle
├── Left Hip Kinematics   (6) : Min, Max, Mean, SD, ROM, Bottom Angle
├── Right Hip Kinematics  (6) : Min, Max, Mean, SD, ROM, Bottom Angle
├── Left Ankle Kinematics (6) : Min, Max, Mean, SD, ROM, Bottom Angle
├── Right Ankle Kinematics(6) : Min, Max, Mean, SD, ROM, Bottom Angle
├── Trunk Kinematics      (6) : Min, Max, Mean, SD, ROM, Excursion
└── Asymmetry Summaries   (3) : Knee Asymmetry Peak, Hip Asymmetry Peak, Ankle Asymmetry Peak

```

---

## Movement Indicators

The pipeline evaluates four descriptive movement indicators to quantify variation across the cohort:

1. **Knee Bilateral Asymmetry (Front View):** Absolute difference between left and right knee planar angles throughout the repetition cycle.
2. **Hip Bilateral Asymmetry (Front View):** Absolute difference between left and right hip angles during movement.
3. **Ankle Bilateral Asymmetry (Front View):** Absolute difference between left and right ankle angle estimates.
4. **Trunk Excursion (Side View):** Total angular variation in forward trunk lean relative to vertical during squat descent and ascent.

---

## Results Snapshot

Summary of descriptive movement statistics derived across all 100 analyzed repetitions per camera view:

| Movement Indicator | View | Cohort Mean | Median | SD | Min | Max | 95th Percentile |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Knee Asymmetry** | Front View | **2.97°** | 2.45° | 2.15° | 0.21° | 13.50° | **11.33°** |
| **Hip Asymmetry** | Front View | **3.75°** | 3.10° | 2.80° | 0.35° | 16.80° | **14.24°** |
| **Ankle Asymmetry** | Front View | **2.07°** | 1.85° | 1.12° | 0.15° | 6.20° | **5.13°** |
| **Trunk Excursion** | Side View | **11.14°** | **9.36°** | **5.90°** | **3.30°** | **29.57°** | **21.75°** |

> **Scientific Interpretation Notice:** These values describe empirical movement distributions in this specific non-gym-trained dataset cohort. The upper 5% percentile cutoffs represent statistical distribution flags within this sample and **should not** be interpreted as clinical injury thresholds or biomechanical pathology markers.

---

## Publication Figures

Select visual figures generated by the automated plotting module (`src/08_generate_figures.py`):

### Side-View Sagittal Kinematics

Time-normalized trunk forward-lean trajectories (101 points) illustrating squat descent, bottom inflection point (~50%), and ascent across all repetitions.

* *Primary plot location:* [`outputs/figures/side_view_trunk_kinematics.png`](https://www.google.com/search?q=outputs/figures/)

### Front-View Left/Right Movement

Bilateral joint trajectories comparing left and right limb kinematics for knees and hips throughout the movement cycle.

* *Primary plot location:* [`outputs/figures/front_view_joint_angles.png`](https://www.google.com/search?q=outputs/figures/)

### Front-View Asymmetry

Distribution plots of absolute bilateral angular differences observed across knee, hip, and ankle joints.

* *Primary plot location:* [`outputs/figures/front_view_asymmetry_distributions.png`](https://www.google.com/search?q=outputs/figures/)

### Movement-Indicator Distributions

Cohort distribution boxplots highlighting sample means, medians, and the upper 95th percentile markers.

* *Primary plot location:* [`outputs/figures/movement_indicator_summary.png`](https://www.google.com/search?q=outputs/figures/)

---

## Reproducibility

Follow these steps to initialize the virtual environment and execute the automated processing pipeline:

### 1. Environment Setup (Windows PowerShell)

```powershell
# Clone repository
git clone https://github.com/nihar787airobotics/squat_dataset_publication.git
cd squat_dataset_publication

# Create virtual environment
python -m venv .venv

# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Install required dependencies
pip install -r requirements.txt

```

### 2. Execution Order

Execute the pipeline scripts sequentially:

```powershell
python src\01_extract_keypoints.py
python src\02_calculate_angles.py
python src\04_segment_squats.py
python src\05_normalize_cycles.py
python src\06_extract_features.py
python src\07_risk_patterns.py
python src\08_generate_figures.py
python src\10_publication_statistics.py

```

---

## Output Mapping

| Processing Script | Direct Output Files | Purpose |
| --- | --- | --- |
| `01_extract_keypoints.py` | `data/processed/keypoints_raw.csv` | Raw 33-landmark coordinate trajectories |
| `02_calculate_angles.py` | `data/processed/joint_angles.csv` | Continuous unsegmented planar joint angles |
| `04_segment_squats.py` | Intermediate repetition boundaries | Repetition start, bottom, and end timestamps |
| `05_normalize_cycles.py` | `data/processed/squat_cycles_normalized.csv` | 101-point resampled kinematic trajectories |
| `06_extract_features.py` | `data/processed/kinematic_features_45.csv` | 45 scalar features per repetition |
| `07_risk_patterns.py` | `outputs/movement_indicators.csv` | Asymmetry and excursion indicator values |
| `08_generate_figures.py` | `outputs/figures/*.png` | Publication figures and distribution plots |
| `10_publication_statistics.py` | `outputs/publication_summary_stats.csv` | Aggregated statistical summary tables |

---

## Scientific Scope and Limitations

When utilizing this dataset, researchers should note the following constraints:

1. **2D Monocular Projections:** Kinematic measures reflect 2D planar projections derived from monocular RGB video frames and do not substitute for 3D optoelectronic motion capture.
2. **Uncalibrated Camera Geometry:** Camera height, elevation angle, and physical camera-to-subject distances were not systematically recorded in absolute physical units.
3. **Cohort Characteristics:** Data were acquired from 20 general, non-gym-trained individuals (15 male, 5 female). Kinematic ranges may differ significantly in trained athletic populations.
4. **Sampling Frame Rates:** Frontal views were processed at ~60 FPS while sagittal views were captured at ~30–60 FPS, necessitating spline interpolation for time-normalization.
5. **Non-Clinical Classification:** Upper 95th percentile indicator cutoffs are empirical statistical descriptors within this sample dataset, not diagnostic medical markers.

---

## Potential Reuse

This dataset and pipeline support research applications in:

* **Computer Vision Benchmarking:** Evaluating keypoint detection stability and landmark jitter across dynamic lower-body exercise tasks.
* **Kinematic Trajectory Analysis:** Applying functional data analysis (FDA) and dynamic time warping (DTW) to time-normalized movement curves.
* **Feature Engineering & ML:** Developing movement quality classification models, rep-counting algorithms, or unsupervised clustering pipelines.
* **Educational Biomechanics:** Providing an open Python codebase for teaching computer-vision-based movement analysis workflows.

---

## Citation

An archival DOI will be registered upon formal publication release. In the interim, please cite this repository as follows:

```bibtex
@misc{nihar787_squat_dataset_2026,
  author       = {Nihar and Contributors},
  title        = {Dual-View Squat Movement Dataset and Kinematic Analysis Pipeline},
  year         = {2026},
  publisher    = {GitHub},
  journal      = {GitHub repository},
  howpublished = {\url{https://github.com/nihar787airobotics/squat_dataset_publication}}
}

```

---

## License

Licensing terms for the dataset and source code are currently being finalized. Please contact the repository maintainer prior to commercial distribution.

---

## Contact

Maintainer: **nihar787airobotics**

Repository Link: [https://github.com/nihar787airobotics/squat_dataset_publication](https://github.com/nihar787airobotics/squat_dataset_publication)
