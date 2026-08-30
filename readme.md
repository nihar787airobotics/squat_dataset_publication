# Dual-View Squat Movement Dataset

A computer-vision dataset and reproducible kinematic analysis pipeline for squat movement analysis.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Pose%20Landmarker%20Full-teal.svg)
![Dataset](https://img.shields.io/badge/Dataset-Open%20Research-green.svg)

This repository provides a dual-view 2D RGB video dataset and an automated Python processing pipeline for extracting planar kinematics from body-weight squat movements. Simultaneous frontal and sagittal video recordings captured across 20 participants yield 100 complete squat repetitions per view. The end-to-end pipeline processes raw video into landmark trajectories, joint angles, time-normalized cycles, 45 engineered features per repetition, and descriptive asymmetry and trunk excursion indicators.

---

## Experimental Setup

![Experimental Setup](docs/images/experimental_setup.png)

*Figure 1: Dual-camera recording configuration in a studio environment.*

---

## Quick Dataset Overview

<table>
  <tr>
    <td align="center" width="16%">
      <font size="6"><b>20</b></font><br/>
      <sub>Participants (15M / 5F)</sub>
    </td>
    <td align="center" width="16%">
      <font size="6"><b>100</b></font><br/>
      <sub>Repetitions per view</sub>
    </td>
    <td align="center" width="16%">
      <font size="6"><b>5</b></font><br/>
      <sub>Repetitions / participant</sub>
    </td>
    <td align="center" width="16%">
      <font size="6"><b>101</b></font><br/>
      <sub>Normalized points / rep</sub>
    </td>
    <td align="center" width="16%">
      <font size="6"><b>45</b></font><br/>
      <sub>Features / repetition</sub>
    </td>
    <td align="center" width="16%">
      <font size="6"><b>2</b></font><br/>
      <sub>Camera views (Front/Side)</sub>
    </td>
  </tr>
</table>

---

## Why This Dataset?

Markerless kinematic assessment using standard RGB camera sensors offers an accessible, non-invasive alternative to optoelectronic motion capture. However, evaluating compound lower-limb exercises like the squat requires multi-perspective observation: sagittal views detail depth and trunk forward lean, while frontal views highlight bilateral asymmetry and dynamic valgus/varus tendencies.

This dataset provides synchronized dual-view video paired with a fully transparent, step-by-step processing pipeline. By executing keypoint detection, joint angle computation, repetition segmentation, and time-normalization, the project bridges raw computer vision outputs with standardized biomechanical representations.

All movement indicators derived by the pipeline—such as frontal angle asymmetries and sagittal trunk excursions—serve as descriptive statistical metrics within this sample. They are structured for algorithmic benchmarking, time-series classification, and feature engineering research.

---

## Processing Pipeline

![Processing Pipeline](docs/images/pipeline.png)

*Figure 2: End-to-end processing pipeline from raw video to publication metrics.*

---

## Model / Pose Estimation

Pose estimation relies on the pre-trained MediaPipe Pose Landmarker Full model (`models/pose_landmarker_full.task`), which predicts 33 3D body landmark locations in normalized image coordinates alongside visibility metrics.

```python
# Pipeline inference configuration
min_pose_detection_confidence = 0.5
min_pose_presence_confidence  = 0.5
min_tracking_confidence       = 0.5
