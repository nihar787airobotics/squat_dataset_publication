#!/usr/bin/env python
"""Manual squat keypoint annotation tool. Run this file without arguments to start.

Default: process the videos in D:/Programming/research/video to be blurred one at a time.
The already annotated WhatsApp clip is skipped. For each other clip, mark rep start,
bottom and end frames, then manually annotate five frames from each marked rep.
Use --single-video to use the original single-video workflow. Matching keypoint
or repetition CSVs select frames automatically; model coordinates stay hidden.
Videos without matching CSVs open a brief manual repetition-boundary screen.
--check validates selected frames/CSV matching without windows or file changes.
Joint annotation: click video = RED DOT; O/X/Space = skip hidden joint;
B = skip both shoulders on this frame; N/P = frames; Enter = finish video;
Q = save and STOP the entire batch. U = undo. 1-8 select a joint to edit.
During repetition review: S/B/E mark start/bottom/end, U undoes, D starts joints.
Boundary review and joint labels both resume after restarting. Backups of
pre-existing annotation CSVs are retained before their first update.
Existing batch folders under person17_manual_annotations or
manual_annotations_remaining_videos are detected automatically beside this
script. New work defaults to manual_annotations_remaining_videos/batch.
Use --output-dir to resume another output root.
--samples-per-rep 13 selects more frames; --all-frames selects the entire video.
Manual labels are never initialized from model predictions.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from datetime import datetime
import math
import os
import re
import statistics
import tempfile
from pathlib import Path

try:
    import cv2
except ImportError as exc:
    raise SystemExit("Install OpenCV first: python -m pip install opencv-python") from exc

BASE = Path(__file__).resolve().parent
DEFAULT_VIDEO = BASE / "WhatsApp Video 2026-10-14 at 11.01.05 PM.mp4"
DEFAULT_VIDEO_DIR = BASE / "data" / "raw" / "front"
COMPLETED_VIDEO_NAME = "WhatsApp Video 2026-10-14 at 11.01.05 PM"
JOINTS = ["left_shoulder", "right_shoulder", "left_hip", "right_hip",
          "left_knee", "right_knee", "left_ankle", "right_ankle"]
SOURCE_FIELDS = ["participant_id", "video_name", "camera_view", "frame_number", "timestamp_seconds",
                 "phase_label", "rep_id", "source_video_path_private", "source_csv_path_private",
                 "source_frames_csv_private", "source_repetitions_csv_private", "frame_width", "frame_height", "video_fps"]
FIELDS = ["participant_id", "video_name", "camera_view", "frame_number", "timestamp_seconds", "phase_label",
          "joint_name", "manual_x", "manual_y", "annotation_status", "annotator_id", "notes",
          "source_csv_path_private", "rep_id", "source_video_path_private", "frame_width", "frame_height",
          "video_fps", "manual_x_normalized", "manual_y_normalized"]
CONVENTION = "LEFT/RIGHT = the participant's own anatomical sides, never screen left/right."


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def backup_once(path):
    """Keep an immutable copy of any pre-existing output before the first update."""
    if path.is_file():
        backup = path.with_name(path.name + ".before_update.bak")
        if not backup.exists():
            shutil.copy2(path, backup)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(data, stream, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def write_csv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".annotation_", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def frame_key(row):
    return row["participant_id"], row["video_name"], int(row["frame_number"])


def video_info(path):
    cap = cv2.VideoCapture(str(path))
    try:
        ok, image = cap.read()
        if not ok:
            raise ValueError(f"Cannot read video: {path}")
        fps, count = float(cap.get(cv2.CAP_PROP_FPS)), round(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not math.isfinite(fps) or fps <= 0 or count <= 0:
            raise ValueError(f"Invalid frame count/FPS: {path}")
        return dict(width=image.shape[1], height=image.shape[0], fps=fps, frames=count)
    finally:
        cap.release()


def sample_rep(start, bottom, end, count):
    left = (count - 1) // 2
    right = count - 1 - left
    return sorted({round(start + (bottom - start) * i / left) for i in range(left + 1)}
                  | {round(bottom + (end - bottom) * i / right) for i in range(right + 1)})


def prepare_frames(args):
    if args.selected_frames:
        rows = read_csv(args.selected_frames)
        required = SOURCE_FIELDS[:6] + ["source_video_path_private", "source_csv_path_private"]
        if not rows or any(any(field not in row for field in required) for row in rows):
            raise ValueError("Invalid selected-frame manifest; use the earlier annotator's manifest schema.")
        info_cache = {}
        for row in rows:
            source = Path(row["source_video_path_private"])
            if not source.is_absolute():
                source = args.selected_frames.resolve().parent / source
            source = source.resolve()
            if source not in info_cache:
                info_cache[source] = video_info(source)
            info = info_cache[source]
            if not 0 <= int(row["frame_number"]) < info["frames"]:
                raise ValueError("Selected frame is outside its source video.")
            row.update({field: row.get(field, "") for field in SOURCE_FIELDS})
            row.update(source_video_path_private=str(source), frame_width=info["width"],
                       frame_height=info["height"], video_fps=info["fps"])
        return rows

    video = args.video.resolve()
    info = video_info(video)
    candidates, reasons = [], []
    for view, prefix in (("side", "side_person17"), ("front", "person_17")):
        if args.view != "auto" and view != args.view:
            continue
        frames_path = args.frames_csv or BASE / f"{prefix}_frames.csv"
        points_path = args.keypoints_csv or BASE / f"{prefix}.csv"
        reps_path = args.repetitions_csv or BASE / f"{prefix}_repetitions.csv"
        if not all(path.is_file() for path in (frames_path, points_path, reps_path)):
            reasons.append(f"{view}: missing CSV input(s)")
            continue
        rows = read_csv(frames_path)
        required = {"video", "view", "frame", "timestamp_ms", "fps", "rep_id", "phase"}
        if not rows or not required.issubset(rows[0]):
            raise ValueError(f"{frames_path} must be a per-frame CSV, not a normalized-cycle CSV.")
        ids = [int(row["frame"]) for row in rows]
        with points_path.open(encoding="utf-8-sig", newline="") as stream:
            points_meta = next(csv.DictReader(stream), None)
        if not points_meta or not {"frame_width", "frame_height", "video", "view"}.issubset(points_meta):
            raise ValueError(f"Missing video/dimension metadata in {points_path}")
        valid = (sorted(ids) == list(range(info["frames"]))
                 and all(row["view"] == view and abs(float(row["fps"]) - info["fps"]) <= .05
                         and abs(float(row["timestamp_ms"]) - int(row["frame"]) * 1000 / info["fps"]) <= 2
                         for row in rows)
                 and int(points_meta["frame_width"]) == info["width"]
                 and int(points_meta["frame_height"]) == info["height"]
                 and points_meta["view"] == view
                 and all(row["video"] == points_meta["video"] for row in rows))
        if valid:
            candidates.append((view, rows, points_path, frames_path, reps_path))
        else:
            reasons.append(f"{view}: exports have {len(rows)} frames at {rows[0]['fps']} FPS; "
                           f"video has {info['frames']} frames at {info['fps']:.6f} FPS")
    if len(candidates) != 1:
        raise ValueError("Cannot uniquely match video and CSVs. " + "; ".join(reasons)
                         + ". Use the correct --video and --view; front/side videos cannot be interchanged.")
    view, frame_rows, points_path, frames_path, reps_path = candidates[0]
    by_frame = {int(row["frame"]): row for row in frame_rows}
    repetitions = read_csv(reps_path)
    if not repetitions or not {"rep_id", "start_frame", "bottom_frame", "end_frame"}.issubset(repetitions[0]):
        raise ValueError(f"Invalid repetitions CSV: {reps_path}")
    ids = [int(rep["rep_id"]) for rep in repetitions]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate repetition IDs.")
    chosen, phase_override, rep_override = set(), {}, {}
    for rep in repetitions:
        rep_id, start, bottom, end = [int(rep[field]) for field in ("rep_id", "start_frame", "bottom_frame", "end_frame")]
        if not 0 <= start <= bottom <= end < info["frames"]:
            raise ValueError(f"Invalid repetition {rep_id} frame boundaries.")
        if args.rep_id is not None and rep_id != args.rep_id:
            continue
        selected = sample_rep(start, bottom, end, args.samples_per_rep)
        chosen.update(selected)
        rep_override.update({index: rep_id for index in selected})
        phase_override.update({start: "descent_start", bottom: "bottom", end: "end_standing"})
    if args.rep_id is not None and args.rep_id not in ids:
        raise ValueError(f"Unknown repetition {args.rep_id}; available: {ids}")
    if args.all_frames:
        chosen = set(by_frame)
    result = []
    for index in sorted(chosen):
        row = by_frame[index]
        result.append(dict(participant_id=args.participant_id, video_name=Path(row["video"]).stem,
                           camera_view=view, frame_number=index, timestamp_seconds=f"{float(row['timestamp_ms']) / 1000:.6f}",
                           phase_label=phase_override.get(index, row["phase"]), rep_id=rep_override.get(index, row["rep_id"]),
                           source_video_path_private=str(video), source_csv_path_private=str(points_path.resolve()),
                           source_frames_csv_private=str(frames_path.resolve()), source_repetitions_csv_private=str(reps_path.resolve()),
                           frame_width=info["width"], frame_height=info["height"], video_fps=info["fps"]))
    if not result:
        raise ValueError("No frames selected.")
    return result


class Annotator:
    HEADER = 276

    def __init__(self, frames, output_dir, annotator_id, max_width=1280, max_height=850):
        self.frames, self.output_dir, self.annotator_id = frames, Path(output_dir), annotator_id
        self.index = self.joint_index = 0
        self.annotations, self.saved_rows, self.history = {}, {}, []
        self.capture = self.capture_path = self.cached_key = self.cached_image = None
        self.max_width, self.max_height = max_width, max_height
        self.scale_x = self.scale_y = 1.
        self.image_width = self.image_height = self.display_width = self.display_height = 0
        self.window = "Manual squat joints - click the VIDEO below the controls"
        self.feedback = "Click a joint on the video, or skip it if hidden."
        self.last_joint = "none"
        self.batch_progress = ""
        self.finished_target = False
        self.buttons = []
        self.session_backup_done = False
        if len({r["participant_id"] for r in frames}) != 1 or len({r["camera_view"] for r in frames}) != 1:
            raise ValueError("Use a separate session for each participant and camera view.")
        self.output_path = self.output_dir / f"manual_squat_keypoint_annotations_{frames[0]['participant_id']}_{frames[0]['camera_view']}_{annotator_id}.csv"
        keys = [frame_key(row) for row in frames]
        if len(set(keys)) != len(keys):
            raise ValueError("Duplicate selected video/frame keys.")
        self.source_lookup = {frame_key(row): row for row in frames}
        self.fieldnames = list(FIELDS)
        self.resume()
        self.finished_target = len(self.annotations.get(self.key(), {})) == len(JOINTS)

    def current_row(self):
        return self.frames[self.index]

    def key(self):
        return frame_key(self.current_row())

    def resume(self):
        if not self.output_path.exists():
            return
        saved = read_csv(self.output_path)
        if saved:
            self.fieldnames += [name for name in saved[0] if name not in self.fieldnames]
        allowed_sources = {os.path.normcase(str(Path(r["source_video_path_private"]).resolve())) for r in self.frames}
        for row in saved:
            key, joint = frame_key(row), row["joint_name"]
            if row["annotator_id"] != self.annotator_id or joint not in JOINTS:
                raise ValueError("Existing output has a different annotator or unknown joint.")
            if key[0] != self.frames[0]["participant_id"] or row["camera_view"] != self.frames[0]["camera_view"]:
                raise ValueError("Existing output belongs to another participant/view.")
            if row.get("source_video_path_private") and os.path.normcase(str(Path(row["source_video_path_private"]).resolve())) not in allowed_sources:
                raise ValueError("Existing annotations refer to another video; choose a different --output-dir.")
            if (key, joint) in self.saved_rows:
                raise ValueError("Duplicate joint records in existing annotations.")
            self.saved_rows[(key, joint)] = row
            status = row["annotation_status"]
            if status == "visible":
                x, y = float(row["manual_x"]), float(row["manual_y"])
                if not math.isfinite(x) or not math.isfinite(y):
                    raise ValueError("Non-finite saved manual coordinates.")
                source = self.source_lookup.get(key)
                if source and not (0 <= x < float(source['frame_width']) and 0 <= y < float(source['frame_height'])):
                    raise ValueError("Saved manual coordinates fall outside the original video.")
                self.annotations.setdefault(key, {})[joint] = (x, y, status)
            elif status == "occluded_or_uncertain":
                self.annotations.setdefault(key, {})[joint] = ("", "", status)
            elif status != "missing":
                raise ValueError(f"Unknown annotation status: {status}")
        for i, row in enumerate(self.frames):
            missing = [j for j, joint in enumerate(JOINTS) if joint not in self.annotations.get(frame_key(row), {})]
            if missing:
                self.index, self.joint_index = i, missing[0]
                break
        print(f"Resumed {self.output_path}")

    def load_frame(self):
        if self.cached_key == self.key():
            return self.cached_image
        row = self.current_row()
        source = str(Path(row["source_video_path_private"]).resolve())
        if source != self.capture_path:
            if self.capture:
                self.capture.release()
            self.capture, self.capture_path = cv2.VideoCapture(source), source
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, int(row["frame_number"]))
        ok, image = self.capture.read()
        if not ok:
            raise RuntimeError(f"Could not read frame {row['frame_number']} from {source}")
        self.image_height, self.image_width = image.shape[:2]
        if row.get("frame_width") and (int(row["frame_width"]), int(row["frame_height"])) != (self.image_width, self.image_height):
            raise ValueError("Video geometry changed; cannot mix coordinate spaces.")
        row.update(frame_width=self.image_width, frame_height=self.image_height, video_fps=self.capture.get(cv2.CAP_PROP_FPS))
        self.cached_key, self.cached_image = self.key(), image
        return image

    def draw(self):
        image = self.load_frame()
        scale = min(self.max_width / self.image_width, (self.max_height - self.HEADER) / self.image_height, 2.)
        self.display_width, self.display_height = round(self.image_width * scale), round(self.image_height * scale)
        self.scale_x, self.scale_y = self.display_width / self.image_width, self.display_height / self.image_height
        display = cv2.copyMakeBorder(cv2.resize(image, (self.display_width, self.display_height)), self.HEADER, 0, 0, 0,
                                     cv2.BORDER_CONSTANT, value=(30, 30, 30))
        if display.shape[1] < 1040:
            display = cv2.copyMakeBorder(display, 0, 0, 0, 1040-display.shape[1],
                                         cv2.BORDER_CONSTANT, value=(30, 30, 30))
        for joint, (x, y, status) in self.annotations.get(self.key(), {}).items():
            if status == "visible":
                px, py = round(x * self.scale_x), self.HEADER + round(y * self.scale_y)
                cv2.circle(display, (px, py), 7, (0, 0, 255), -1)
                cv2.putText(display, joint, (px + 6, py - 5), cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 0, 255), 1)
        row = self.current_row()
        j = self.joint_index
        target = "FRAME COMPLETE - N for next" if self.finished_target else JOINTS[j].upper()
        previous = JOINTS[j-1] if j > 0 else "none"
        following = JOINTS[j+1] if j < 7 else "end of frame"
        lines = [f"{self.batch_progress} | {row['participant_id']} {row['camera_view'].upper()} | {row['video_name']}",
                 f"Frame {self.index+1}/{len(self.frames)} (video frame {row['frame_number']}) | Rep {row.get('rep_id', '?')} | {len(self.annotations.get(self.key(), {}))}/8 reviewed",
                 f"CURRENT: {target}",
                 f"PREVIOUS: {previous} | NEXT: {following}",
                 f"LAST: {self.last_joint} | {self.feedback}",
                 CONVENTION,
                 "Click video=red dot | O/X/Space=skip hidden | B=skip shoulders | U=undo",
                 "N/P=next/previous frame | Tab=unfinished | 1-8/J/K=choose joint",
                 "Enter=finish video when complete | S=save | Q/Esc=save and STOP batch"]
        for i, text in enumerate(lines):
            # Keep controls usable for narrow portrait videos as well.
            cv2.putText(display, text, (8, 19 + i * 23), cv2.FONT_HERSHEY_SIMPLEX, .43,
                        (0, 0, 255) if i in (2, 3, 4) else (235, 235, 235), 1, cv2.LINE_AA)
        self.buttons = [(8, 222, 202, 266, "skip", "Skip invisible (O)"),
                        (210, 222, 428, 266, "shoulders", "Skip shoulders (B)"),
                        (436, 222, 632, 266, "next", "Next frame (N)")]
        for x1, y1, x2, y2, _, label in self.buttons:
            cv2.rectangle(display, (x1, y1), (x2, y2), (70, 70, 70), -1)
            cv2.putText(display, label, (x1+6, y1+27), cv2.FONT_HERSHEY_SIMPLEX, .47, (255, 255, 255), 1)
        return display

    def complete(self):
        return all(len(self.annotations.get(frame_key(row), {})) == len(JOINTS) for row in self.frames)

    def skip_shoulders(self):
        for j in (0, 1):
            if JOINTS[j] not in self.annotations.get(self.key(), {}):
                self.joint_index, self.finished_target = j, False
                self.record(("", "", "occluded_or_uncertain"))
        missing = [i for i, joint in enumerate(JOINTS) if joint not in self.annotations.get(self.key(), {})]
        self.finished_target = not missing
        if missing:
            self.joint_index = missing[0]

    def record(self, value):
        if self.finished_target and value is not None:
            self.feedback = "Frame complete. Press N, or choose 1-8 to edit."
            return
        key, joint = self.key(), JOINTS[self.joint_index]
        self.history.append((self.index, self.joint_index, self.annotations.get(key, {}).get(joint)))
        if value is None:
            self.annotations.get(key, {}).pop(joint, None)
            self.finished_target = False
            self.feedback = "Cleared selected joint."
        else:
            self.annotations.setdefault(key, {})[joint] = value
            self.last_joint = joint + (" [RED DOT]" if value[2] == "visible" else " [OCCLUDED]")
            self.feedback = "Saved."
            missing = [i for i, name in enumerate(JOINTS) if name not in self.annotations.get(key, {})]
            self.finished_target = not missing
            later = [i for i in missing if i > self.joint_index]
            if missing:
                self.joint_index = (later or missing)[0]
        self.save(quiet=True)

    def on_mouse(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        for x1, y1, x2, y2, action, _ in self.buttons:
            if x1 <= x <= x2 and y1 <= y <= y2:
                if action == "skip":
                    self.record(("", "", "occluded_or_uncertain"))
                elif action == "shoulders":
                    self.skip_shoulders()
                else:
                    self.navigate(1)
                return
        if not (0 <= x < self.display_width and self.HEADER <= y < self.HEADER + self.display_height):
            return
        # Header/scaling affect display coordinates, never the saved pixel coordinates.
        ox, oy = x / self.scale_x, (y - self.HEADER) / self.scale_y
        self.record((min(self.image_width - 1., ox), min(self.image_height - 1., oy), "visible"))

    def undo(self):
        if not self.history:
            return
        self.index, self.joint_index, previous = self.history.pop()
        joint = JOINTS[self.joint_index]
        self.finished_target = False
        self.feedback = "Undid last action."
        if previous is None:
            self.annotations.get(self.key(), {}).pop(joint, None)
        else:
            self.annotations.setdefault(self.key(), {})[joint] = previous
        self.save(quiet=True)

    def navigate(self, offset):
        self.index = max(0, min(len(self.frames) - 1, self.index + offset))
        missing = [i for i, joint in enumerate(JOINTS) if joint not in self.annotations.get(self.key(), {})]
        self.joint_index = missing[0] if missing else 7
        self.finished_target = not missing

    def next_unfinished(self):
        for offset in range(1, len(self.frames) + 1):
            target = (self.index + offset) % len(self.frames)
            missing = [i for i, joint in enumerate(JOINTS) if joint not in self.annotations.get(frame_key(self.frames[target]), {})]
            if missing:
                self.index, self.joint_index = target, missing[0]
                self.finished_target = False
                return

    def save(self, quiet=False):
        # Preserve annotations outside the selected frame subset if sampling changes.
        rows = [row for (key, _), row in self.saved_rows.items() if key not in self.source_lookup]
        for source in self.frames:
            key = frame_key(source)
            width, height = float(source["frame_width"]), float(source["frame_height"])
            for joint in JOINTS:
                x, y, status = self.annotations.get(key, {}).get(joint, ("", "", "missing"))
                row = dict(self.saved_rows.get((key, joint), {}))
                row.update({name: source.get(name, "") for name in SOURCE_FIELDS if name in self.fieldnames})
                row.update(joint_name=joint, manual_x=x, manual_y=y, annotation_status=status, annotator_id=self.annotator_id,
                           manual_x_normalized=x / width if status == "visible" else "",
                           manual_y_normalized=y / height if status == "visible" else "")
                if not row.get("notes"):
                    row["notes"] = "side_convention=participant_anatomical; coordinates=original_video_pixels; model_points_not_shown_by_this_tool"
                rows.append(row)
        if not self.session_backup_done:
            backup_once(self.output_path)
            self.session_backup_done = True
        write_csv(self.output_path, self.fieldnames, rows)
        if not quiet:
            print(f"Saved {self.output_path}")

    def close(self):
        if self.capture:
            self.capture.release()
        self.capture = None

    def run(self):
        if os.name == "nt":
            import ctypes
            try:
                ctypes.windll.user32.SetProcessDPIAware()
            except OSError:
                pass
        cv2.namedWindow(self.window, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.window, self.on_mouse)
        try:
            while True:
                cv2.imshow(self.window, self.draw())
                key = cv2.waitKey(40) & 0xFF
                try:
                    visible = cv2.getWindowProperty(self.window, cv2.WND_PROP_VISIBLE) >= 1
                except cv2.error:
                    visible = False
                if key in (ord("q"), ord("Q"), 27) or not visible:
                    return "quit"
                if 65 <= key <= 90:
                    key += 32
                if key in (10, 13) or (key == ord("n") and self.index == len(self.frames)-1):
                    if self.complete():
                        return "complete"
                    self.feedback = "Some slots are unfinished. Tab finds them; O marks hidden joints."
                    continue
                if key == ord("s"):
                    self.save()
                elif key in (ord("n"), ord("p")):
                    self.navigate(1 if key == ord("n") else -1)
                elif key == 9:
                    self.next_unfinished()
                elif key in (ord("j"), ord("k")):
                    self.finished_target = False
                    self.joint_index = min(len(JOINTS)-1, max(0, self.joint_index + (1 if key == ord("k") else -1)))
                elif ord("1") <= key <= ord("8"):
                    self.finished_target = False
                    self.joint_index = key - ord("1")
                elif key == ord("b"):
                    self.skip_shoulders()
                elif key in (ord("o"), ord("x"), 32):
                    self.record(("", "", "occluded_or_uncertain"))
                elif key == ord("u"):
                    self.undo()
                elif key in (8, 127):
                    self.record(None)
        finally:
            try:
                self.save()
            finally:
                self.close()
                cv2.destroyAllWindows()


def video_identity(path):
    """Get participant/view from the known person_17 and side_person17 naming."""
    stem = re.sub(r"_face_blurred$", "", path.stem, flags=re.IGNORECASE)
    match = re.search(r"(?:^|_)(?:side_person_?|frontperson_?|person_?)(\d+)(?:_|$)", stem, re.IGNORECASE)
    if not match:
        return None
    participant = f"P{int(match.group(1)):03d}"
    view = "side" if stem.lower().startswith("side_person") else "front"
    return participant, view, stem


def mark_repetitions(video_path, participant, view, title, args):
    """Let the annotator mark rep start/bottom/end frames without model overlays."""
    info = video_info(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    frame_number = 0
    stage = 0  # start -> bottom -> end
    current = []
    repetitions = []
    checkpoint = args.output_dir / "batch" / title / "repetition_review.json"
    if checkpoint.is_file():
        data = json.loads(checkpoint.read_text(encoding="utf-8"))
        if data.get("video") != str(video_path.resolve()) or data.get("info") != info:
            raise ValueError(f"Review checkpoint does not match video: {checkpoint}")
        repetitions = [tuple(rep) for rep in data["repetitions"]]
        current = list(data["current"])
        stage, frame_number = len(current), int(data["frame"])
    window = f"Mark squat repetitions: {title}"
    print(f"\nReviewing {title}. LEFT/RIGHT are the participant's anatomical sides.")
    print("Use N/P to move one frame, arrows to jump 15 frames. Mark S=start, B=bottom, E=end.")
    print("Mark S/B/E in order. U undoes a boundary. D starts joints. Q saves and stops.")
    pending = {"key": None}
    review_buttons = [(8, 112, 158, 154, ord("s"), "START (S)"),
                      (166, 112, 316, 154, ord("b"), "BOTTOM (B)"),
                      (324, 112, 474, 154, ord("e"), "END (E)"),
                      (482, 112, 742, 154, ord("d"), "START JOINTS (D)")]
    def review_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            for x1, y1, x2, y2, key, _ in review_buttons:
                if x1 <= x <= x2 and y1 <= y <= y2:
                    pending["key"] = key
    last_state = None
    try:
        cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(window, review_click)
        cv2.createTrackbar("Seek frame", window, frame_number, info["frames"]-1,
                           lambda value: pending.update(seek=value))
        while 0 <= frame_number < info["frames"]:
            state = dict(video=str(video_path.resolve()), info=info,
                         repetitions=list(repetitions), current=list(current), frame=frame_number)
            if state != last_state:
                write_json(checkpoint, state)
                last_state = state
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ok, frame = cap.read()
            if not ok:
                break
            h, w = frame.shape[:2]
            scale = min(args.max_width / w, (args.max_height - 170) / h, 1.5)
            display = cv2.resize(frame, (round(w * scale), round(h * scale)))
            labels = [("S", "start"), ("B", "bottom"), ("E", "end")]
            stage_text = labels[stage][1] if stage < len(labels) else "finish"
            completed = len(repetitions)
            display = cv2.copyMakeBorder(display, 164, 0, 0, max(0, 1040-display.shape[1]), cv2.BORDER_CONSTANT, value=(30, 30, 30))
            messages = [f"{participant} | {view.upper()} | {title} | frame {frame_number}/{info['frames']-1}",
                        f"Rep {completed+1}: mark {stage_text.upper()} | completed repetitions: {completed}",
                        "REVIEW: S/B/E or buttons = boundaries | N/P seek | U undo | D joints | Q save/STOP"]
            for i, message in enumerate(messages):
                cv2.putText(display, message, (10, 25 + i * 29), cv2.FONT_HERSHEY_SIMPLEX, .54,
                            (0, 230, 255) if i == 1 else (245, 245, 245), 1, cv2.LINE_AA)
            for x1, y1, x2, y2, _, label in review_buttons:
                cv2.rectangle(display, (x1, y1), (x2, y2), (70, 70, 70), -1)
                cv2.putText(display, label, (x1+6, y1+27), cv2.FONT_HERSHEY_SIMPLEX, .5, (255,255,255), 1)
            cv2.imshow(window, display)
            key = cv2.waitKeyEx(40)
            seek = pending.pop("seek", None)
            if seek is not None:
                frame_number = seek
            if pending["key"] is not None:
                key, pending["key"] = pending["key"], None
            if key in (ord("q"), ord("Q"), 27):
                return None
            try:
                if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    return None
            except cv2.error:
                return None
            if key in (ord("u"), ord("U")):
                if current:
                    frame_number = current.pop()
                elif repetitions:
                    current = list(repetitions.pop())
                    frame_number = current.pop()
                stage = len(current)
            elif key in (ord("n"), ord("N")):
                frame_number = min(frame_number + 1, info["frames"] - 1)
            elif key in (ord("p"), ord("P")):
                frame_number = max(frame_number - 1, 0)
            elif key in (2424832, 65361):  # left arrow on Windows/OpenCV and X11
                frame_number = max(frame_number - 15, 0)
            elif key in (2555904, 65363):  # right arrow
                frame_number = min(frame_number + 15, info["frames"] - 1)
            elif key in (ord("s"), ord("S"), ord("b"), ord("B"), ord("e"), ord("E")):
                requested = {ord("s"): 0, ord("b"): 1, ord("e"): 2,
                             ord("S"): 0, ord("B"): 1, ord("E"): 2}[key]
                if requested != stage:
                    print(f"Mark {labels[stage][1]} next; ignored the out-of-order key.")
                    continue
                if current and frame_number <= current[-1]:
                    print("Choose a later frame for each phase in the repetition.")
                    continue
                if stage == 0 and repetitions and frame_number <= repetitions[-1][2]:
                    print("Start the next repetition after the previous end frame.")
                    continue
                current.append(frame_number)
                stage += 1
                if stage == 3:
                    repetitions.append(tuple(current))
                    current, stage = [], 0
                    print(f"Recorded repetition {len(repetitions)}: {repetitions[-1]}.")
            elif key in (ord("d"), ord("D")):
                if current:
                    print("Finish marking this repetition's bottom and end before continuing.")
                elif not repetitions:
                    print("Mark at least one complete repetition first.")
                else:
                    print(f"Using {len(repetitions)} manually marked repetitions.")
                    return repetitions
            if cv2.getTrackbarPos("Seek frame", window) != frame_number:
                cv2.setTrackbarPos("Seek frame", window, frame_number)
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return None


def frames_from_marked_repetitions(video_path, participant, view, title, repetitions, count,
                                   source_csv="", source_repetitions_csv=""):
    info = video_info(video_path)
    result = []
    for rep_id, (start, bottom, end) in enumerate(repetitions, 1):
        if not 0 <= start < bottom < end < info["frames"]:
            raise ValueError(f"Invalid marked boundaries for repetition {rep_id} in {title}.")
        sampled = sample_rep(start, bottom, end, count)
        if len(sampled) != count:
            raise ValueError(f"Repetition {rep_id} is too short for {count} distinct representative frames.")
        if result and start <= int(result[-1]["frame_number"]):
            raise ValueError("Repetitions overlap or share an endpoint; review their boundaries.")
        for frame in sampled:
            if frame == start:
                phase = "descent_start"
            elif frame == bottom:
                phase = "bottom"
            elif frame == end:
                phase = "end_standing"
            elif frame < bottom:
                phase = "descent"
            else:
                phase = "ascent"
            result.append(dict(participant_id=participant, video_name=title, camera_view=view,
                               frame_number=frame, timestamp_seconds=f"{frame / info['fps']:.6f}",
                               phase_label=phase, rep_id=rep_id,
                               source_video_path_private=str(video_path.resolve()), source_csv_path_private=source_csv,
                               source_frames_csv_private="", source_repetitions_csv_private=source_repetitions_csv,
                               frame_width=info["width"], frame_height=info["height"], video_fps=info["fps"]))
    return result


def matching_prediction_csv(video_path, title, view, folders):
    """Find a normalized per-landmark CSV for this exact clip and camera view."""
    video = video_info(video_path)
    candidates = []
    seen = set()
    for folder in folders:
        if not folder.is_dir():
            continue
        for path in folder.glob("*.csv"):
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            try:
                with path.open(encoding="utf-8-sig", newline="") as stream:
                    reader = csv.DictReader(stream)
                    fields = set(reader.fieldnames or ())
                    if not {"video", "view", "frame", "timestamp_ms", "fps", "frame_width", "frame_height", "landmark_name", "x", "y"}.issubset(fields):
                        continue
                    first = next(reader, None)
                if not first or clip_name(first["video"]) != clip_name(title) or first["view"].casefold() != view:
                    continue
                csv_fps = float(first["fps"])
                if (int(first["frame_width"]) != video["width"] or int(first["frame_height"]) != video["height"]
                        or abs(csv_fps - video["fps"]) > .05):
                    print(f"CSV metadata differs from {video_path.name}; won't use {path.name} for frame selection.")
                    continue
                candidates.append(path.resolve())
            except (OSError, KeyError, ValueError, StopIteration, UnicodeError):
                continue
    if len(candidates) > 1:
        raise ValueError(f"More than one matching keypoint CSV for {video_path.name}: {candidates}")
    return candidates[0] if candidates else None


def detect_repetitions_from_keypoints(video_path, title, prediction_csv):
    """Use the matching CSV's knee-angle curve only to choose representative frames."""
    info = video_info(video_path)
    by_frame = {}
    with prediction_csv.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            validate_metadata(row, video_path, title, video_identity(video_path)[1], info)
            frame = int(row["frame"])
            if not 0 <= frame < info["frames"]:
                raise ValueError(f"CSV frame {frame} is outside {video_path.name}.")
            points = by_frame.setdefault(frame, {})
            if row["landmark_name"] in points:
                raise ValueError("Duplicate frame/landmark in model CSV.")
            points[row["landmark_name"]] = row
    if set(by_frame) != set(range(info["frames"])):
        raise ValueError(f"The matching keypoint CSV does not cover every frame of {video_path.name}.")

    def knee_angle(points, side):
        names = [f"{side}_{part}" for part in ("hip", "knee", "ankle")]
        try:
            rows = [points[name] for name in names]
            if any(float(row.get("visibility", "1") or 0) < .2 for row in rows):
                return math.nan
            # This export uses normalized x/y: restore aspect ratio for angle selection.
            xy = [(float(row["x"]) * info["width"], float(row["y"]) * info["height"]) for row in rows]
            if any(not math.isfinite(v) for point in xy for v in point):
                return math.nan
        except (KeyError, TypeError, ValueError):
            return math.nan
        a, b, c = xy
        u, v = (a[0]-b[0], a[1]-b[1]), (c[0]-b[0], c[1]-b[1])
        denominator = math.hypot(*u) * math.hypot(*v)
        if not denominator:
            return math.nan
        cosine = max(-1., min(1., (u[0]*v[0] + u[1]*v[1]) / denominator))
        return math.degrees(math.acos(cosine))

    angles = []
    for frame in range(info["frames"]):
        values = [knee_angle(by_frame[frame], side) for side in ("left", "right")]
        values = [value for value in values if math.isfinite(value)]
        angles.append(statistics.median(values) if values else math.nan)
    valid = [i for i, value in enumerate(angles) if math.isfinite(value)]
    if len(valid) < info["fps"]:
        raise ValueError(f"Not enough usable knee-angle frames in {prediction_csv.name}.")
    # Interpolate short gaps in a landmark track before smoothing the angle trace.
    for index, value in enumerate(angles):
        if math.isfinite(value):
            continue
        before = next((i for i in range(index-1, -1, -1) if math.isfinite(angles[i])), None)
        after = next((i for i in range(index+1, len(angles)) if math.isfinite(angles[i])), None)
        if before is not None and after is not None and after-before <= max(2, round(info["fps"]*.35)):
            fraction = (index-before) / (after-before)
            angles[index] = angles[before] + fraction * (angles[after]-angles[before])
    smooth_radius = max(1, round(info["fps"] * .06))
    smooth = [statistics.median(angles[max(0, i-smooth_radius):min(len(angles), i+smooth_radius+1)])
              if math.isfinite(angles[i]) else math.nan for i in range(len(angles))]
    active = [math.isfinite(value) and value < 140 for value in smooth]
    max_gap = max(1, round(info["fps"] * .35))
    # Join brief angle-tracking gaps inside one descent/bottom/ascent cycle.
    lows = [i for i, value in enumerate(active) if value]
    groups = []
    for index in lows:
        if groups and index - groups[-1][-1] <= max_gap + 1:
            groups[-1].append(index)
        else:
            groups.append([index])
    repetitions = []
    minimum_active = max(3, round(info["fps"] * .20))
    for group in groups:
        if len(group) < minimum_active:
            continue
        first, last = group[0], group[-1]
        bottom = min(range(first, last+1), key=lambda i: smooth[i] if math.isfinite(smooth[i]) else 180.)
        if smooth[bottom] >= 130:
            continue
        start, end = first, last
        while start > 0 and math.isfinite(smooth[start-1]) and smooth[start-1] < 150:
            start -= 1
        while end + 1 < len(smooth) and math.isfinite(smooth[end+1]) and smooth[end+1] < 150:
            end += 1
        if not repetitions or start > repetitions[-1][2]:
            repetitions.append((start, bottom, end))
    if not repetitions:
        raise ValueError(f"No clear squat repetitions detected from {prediction_csv.name}; use the manual start/bottom/end review.")
    print(f"Frame selection uses {prediction_csv.name} to find {len(repetitions)} squat cycles. Model points stay hidden in the annotation window.")
    return repetitions


def clip_name(value):
    # Windows CSV paths also work during validation on another OS.
    return re.sub(r"_face_blurred$", "", Path(str(value).replace("\\", "/")).stem, flags=re.I).casefold()


def validate_metadata(row, video_path, title, view, info):
    for field in ("video", "video_name"):
        if row.get(field) and clip_name(row[field]) != clip_name(title):
            raise ValueError(f"Video name mismatch in CSV for {video_path.name}")
    for field in ("view", "camera_view"):
        if row.get(field) and row[field].casefold() != view:
            raise ValueError(f"Camera view mismatch for {video_path.name}")
    for field, expected in (("frame_width", info["width"]), ("frame_height", info["height"])):
        if row.get(field) and float(row[field]) != expected:
            raise ValueError(f"{field} mismatch for {video_path.name}")
    for field in ("fps", "video_fps"):
        if row.get(field):
            value = float(row[field])
            if not math.isfinite(value) or abs(value-info["fps"]) > .05:
                raise ValueError(f"FPS mismatch for {video_path.name}")
    if row.get("frame"):
        frame = int(row["frame"])
        if not 0 <= frame < info["frames"]:
            raise ValueError("CSV frame is outside video")
        if row.get("timestamp_ms"):
            value = float(row["timestamp_ms"])
            if not math.isfinite(value) or abs(value-frame*1000/info["fps"]) > 2:
                raise ValueError("CSV timestamp/frame mismatch")


def repetitions_from_csv(video_path, title, folders):
    """Accept exact-name repetition boundaries or a per-frame rep/phase export.

    Optional metadata must agree on EVERY row. Missing metadata is explicitly
    reported; it is never fabricated. Ambiguous matches fall back to review.
    """
    info = video_info(video_path)
    view = video_identity(video_path)[1]
    matches = []
    seen = set()
    for folder in folders:
        if not folder.is_dir():
            continue
        for path in sorted(folder.glob("*.csv")):
            if path.resolve() in seen:
                continue
            seen.add(path.resolve())
            with path.open(encoding="utf-8-sig", newline="") as stream:
                reader = csv.DictReader(stream)
                fields = set(reader.fieldnames or ())
                boundary = {"start_frame", "bottom_frame", "end_frame"}.issubset(fields)
                per_frame = {"frame", "rep_id", "phase"}.issubset(fields) and "landmark_name" not in fields
                if not (boundary or per_frame):
                    continue
                rows = list(reader)
            stem = re.sub(r"_(repetitions|frames)$", "", path.stem, flags=re.I)
            named = clip_name(stem) == clip_name(title)
            if not rows or (not named and not any(clip_name(r.get("video", r.get("video_name", ""))) == clip_name(title) for r in rows)):
                continue
            for row in rows:
                validate_metadata(row, video_path, title, view, info)
            if not {"frame_width", "frame_height", "fps"}.issubset(fields):
                print(f"{path.name}: matching name/boundaries; dimensions or FPS not exported (cannot verify missing metadata).")
            result = []
            if boundary:
                result = [(int(r["start_frame"]), int(r["bottom_frame"]), int(r["end_frame"])) for r in rows]
                if "rep_id" in fields and len({r["rep_id"] for r in rows}) != len(rows):
                    raise ValueError("Duplicate repetition IDs")
            else:
                groups = {}
                if len({int(r["frame"]) for r in rows}) != len(rows):
                    raise ValueError("Duplicate frame in per-frame export")
                for row in rows:
                    if row["rep_id"].strip().casefold() not in ("", "0", "-1", "none", "nan"):
                        groups.setdefault(row["rep_id"], []).append(row)
                for group in groups.values():
                    ids = sorted(int(r["frame"]) for r in group)
                    bottoms = [int(r["frame"]) for r in group if "bottom" in r["phase"].casefold()]
                    if not bottoms:
                        raise ValueError("Per-frame export has no bottom phase; manual review required")
                    result.append((ids[0], sorted(bottoms)[len(bottoms)//2], ids[-1]))
            result.sort()
            if not result or any(not 0 <= a < b < c < info["frames"] for a, b, c in result):
                raise ValueError(f"Invalid repetition boundaries in {path.name}")
            if any(result[i][0] <= result[i-1][2] for i in range(1, len(result))):
                raise ValueError(f"Overlapping repetitions in {path.name}; review manually")
            matches.append((path.resolve(), result, boundary))
    # Explicit boundary files take precedence over per-frame phase files.
    preferred = [item for item in matches if item[2]] or matches
    if len(preferred) > 1:
        raise ValueError(f"Ambiguous CSV matches for {video_path.name}; review manually")
    return preferred[0][:2] if preferred else None



def validated_manifest(path, video, participant, view):
    frames = read_csv(path)
    info = video_info(video)
    if not frames:
        raise ValueError(f"Empty selected-frame manifest: {path}")
    for row in frames:
        if not set(SOURCE_FIELDS).issubset(row):
            raise ValueError(f"Incomplete selected-frame manifest schema: {path}")
        if Path(row['source_video_path_private']).resolve() != video.resolve():
            raise ValueError(f"Manifest points to another video: {path}")
        if row['participant_id'] != participant or row['camera_view'] != view:
            raise ValueError(f"Participant/view mismatch in {path}")
        validate_metadata(row, video, video_identity(video)[2], view, info)
        if not 0 <= int(row['frame_number']) < info['frames']:
            raise ValueError(f"Manifest frame outside video: {path}")
        if abs(float(row['timestamp_seconds']) - int(row['frame_number'])/info['fps']) > .002:
            raise ValueError(f"Manifest timestamp mismatch: {path}")
    if len({frame_key(r) for r in frames}) != len(frames):
        raise ValueError(f"Duplicate selected frames: {path}")
    return frames


def audit_whatsapp(args):
    """Read-only audit: never create a session or save labels for the skipped clip."""
    roots = [args.output_dir, BASE / 'person17_manual_annotations',
             BASE / 'manual_annotations_remaining_videos', BASE]
    found, seen = [], set()
    for root in roots:
        if not root.is_dir():
            continue
        for path in root.rglob('manual_squat_keypoint_annotations*.csv'):
            if path.resolve() in seen:
                continue
            seen.add(path.resolve())
            try:
                rows = [r for r in read_csv(path) if clip_name(r.get('video_name', '')) == clip_name(COMPLETED_VIDEO_NAME)
                        or clip_name(r.get('source_video_path_private', '')) == clip_name(COMPLETED_VIDEO_NAME)]
                if rows:
                    known = {(r.get('frame_number'), r.get('joint_name')): r for r in rows}
                    frames = {r.get('frame_number') for r in rows}
                    missing = sum((f, j) not in known or known[(f,j)].get('annotation_status') not in ('visible', 'occluded_or_uncertain')
                                  for f in frames for j in JOINTS)
                    found.append(f'{path}: {missing} missing slots across {len(frames)} saved frames (read only)')
            except (OSError, ValueError, UnicodeError) as exc:
                found.append(f'{path}: audit could not read file: {exc}')
    return found or ['WhatsApp annotation file not found in the output/script folders; missing slots cannot be verified.']


def run_video_directory(args):
    folder = args.video_dir.resolve()
    if not folder.is_dir():
        raise ValueError(f'Video directory does not exist: {folder}')
    videos = sorted((p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in
                     {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.m4v'}), key=lambda p: p.name.casefold())
    if args.participants:
        wanted = {f"P{number:03d}" for number in args.participants}
        videos = [p for p in videos if (identity := video_identity(p)) is not None and identity[0] in wanted]
    if not videos:
        scope = f" for participants {args.participants}" if args.participants else ""
        raise ValueError(f'No video files found in {folder}{scope}')
    csv_folders = list(dict.fromkeys([folder, args.csv_dir.resolve()]))
    report = []
    stopped = False
    print(f'Videos found: {len(videos)}. Output root: {args.output_dir.resolve()}')
    print('Joints: click VIDEO for red dot; O skips hidden; B skips shoulders; Q saves and stops ALL videos.')
    try:
        for position, video in enumerate(videos, 1):
            entry = dict(video=video.name, status='not reached', output='', incomplete_slots='unknown', detail='')
            report.append(entry)
            if video.stem.casefold() == COMPLETED_VIDEO_NAME.casefold():
                entry.update(status='skipped', detail='WhatsApp clip protected; existing files untouched')
                continue
            identity = video_identity(video)
            if identity is None:
                entry.update(status='skipped', detail='Cannot identify participant/view from filename')
                continue
            participant, view, title = identity
            per_video_dir = args.output_dir / 'batch' / title
            manifest = per_video_dir / f'selected_frames_{participant}_{view}.csv'
            output = per_video_dir / f'manual_squat_keypoint_annotations_{participant}_{view}_{args.annotator_id}.csv'
            entry['output'] = str(output.resolve())
            if stopped:
                entry['detail'] = 'Batch stopped by user; rerun to continue'
                continue
            annotator = None
            try:
                print(f'\nVIDEO {position}/{len(videos)}: {video.name}')
                frames = validated_manifest(manifest, video, participant, view) if manifest.is_file() else None
                if output.exists() and frames is None:
                    raise ValueError('Annotations exist without manifest; preserving files. Restore the manifest before resuming.')
                if frames is None:
                    prediction_csv, rep_export, repetitions = None, None, None
                    try:
                        rep_export = repetitions_from_csv(video, title, csv_folders)
                        if rep_export:
                            repetitions = rep_export[1]
                        else:
                            prediction_csv = matching_prediction_csv(video, title, view, csv_folders)
                            if prediction_csv:
                                repetitions = detect_repetitions_from_keypoints(video, title, prediction_csv)
                        if repetitions:
                            # Reject short/overlapping cycles before committing a manifest.
                            frames = frames_from_marked_repetitions(video, participant, view, title, repetitions,
                                        args.samples_per_rep, str(prediction_csv or ''), str(rep_export[0]) if rep_export else '')
                    except (ValueError, KeyError, UnicodeError) as exc:
                        entry['detail'] = f'CSV matching rejected: {exc}; manual review required'
                        print(entry['detail'])
                        repetitions = frames = None
                    if frames is None:
                        if args.check:
                            entry.update(status='manual review required', detail=entry['detail'] or 'No usable CSV repetition selection')
                            continue
                        print('REPETITION REVIEW (not joint annotation): S=start, B=bottom, E=end; N/P seek; U=undo; D=annotate.')
                        repetitions = mark_repetitions(video, participant, view, title, args)
                        if repetitions is None:
                            entry.update(status='review paused', detail='Boundary progress saved; Q stops the batch')
                            stopped = True
                            continue
                        frames = frames_from_marked_repetitions(video, participant, view, title, repetitions, args.samples_per_rep)
                    if not args.check:
                        # Only a new manifest is written here; an existing one is authoritative.
                        write_csv(manifest, SOURCE_FIELDS, frames)
                annotator = Annotator(frames, per_video_dir, args.annotator_id, args.max_width, args.max_height)
                annotator.batch_progress = f'VIDEO {position}/{len(videos)}'
                entry['incomplete_slots'] = sum(len(JOINTS)-len(annotator.annotations.get(frame_key(r), {})) for r in frames)
                if args.check:
                    for i in range(len(frames)):
                        annotator.index = i
                        annotator.load_frame()
                    entry.update(status='checked', detail=f'{len(frames)} selected frames decoded; no files changed')
                elif annotator.complete():
                    entry.update(status='skipped complete', detail='Every selected slot already reviewed; no annotations changed')
                else:
                    outcome = annotator.run()
                    entry['incomplete_slots'] = sum(len(JOINTS)-len(annotator.annotations.get(frame_key(r), {})) for r in frames)
                    entry['status'] = 'completed' if annotator.complete() else 'paused incomplete'
                    if outcome != 'complete':
                        stopped = True
                        entry['detail'] = 'Q/window close stops batch. Rerun to resume.'
            except (OSError, ValueError, RuntimeError, KeyError, cv2.error) as exc:
                entry.update(status='error - preserved', detail=str(exc))
                print(f'ERROR for {video.name}: {exc}')
                # Stop on errors so the user can fix the current video before proceeding.
                stopped = True
            finally:
                if annotator is not None:
                    annotator.close()
    finally:
        audit = audit_whatsapp(args)
        print('\nSESSION REPORT')
        for entry in report:
            print(f"{entry['status']}: {entry['video']} | incomplete slots: {entry['incomplete_slots']} | {entry['output']} | {entry['detail']}")
        for line in audit:
            print(line)
        if not args.check:
            stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            report_path = args.output_dir / f'progress_report_{stamp}.json'
            write_json(report_path, dict(videos=report, whatsapp_audit=audit, stopped=stopped))
            print(f'Report saved: {report_path.resolve()}')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video-dir", type=Path, default=DEFAULT_VIDEO_DIR,
                        help="Folder processed one video at a time (default: project data/raw/front).")
    parser.add_argument("--csv-dir", type=Path, default=BASE / "data" / "processed" / "segments" / "front",
                        help="Folder searched for matching frame, repetition and keypoint CSVs (default: front segment exports).")
    parser.add_argument("--single-video", action="store_true", help="Use the original one-video workflow.")
    parser.add_argument("--video", type=Path, help="Video path; implies --single-video.")
    parser.add_argument("--view", choices=("auto", "front", "side"), default="auto")
    parser.add_argument("--participant-id", default="P017")
    parser.add_argument("--participants", type=int, nargs="+", metavar="PERSON",
                        help="Batch only these participant numbers, e.g. --participants 17 18 19.")
    parser.add_argument("--annotator-id", default="A01")
    parser.add_argument("--output-dir", type=Path, help="Existing output root to resume; otherwise automatically detect earlier batch folders.")
    parser.add_argument("--frames-csv", type=Path)
    parser.add_argument("--keypoints-csv", type=Path)
    parser.add_argument("--repetitions-csv", type=Path)
    parser.add_argument("--selected-frames", type=Path, help="Optional selected-frame manifest in the earlier annotator format.")
    parser.add_argument("--samples-per-rep", type=int, default=5)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--rep-id", type=int)
    selection.add_argument("--all-frames", action="store_true", help="Annotate every original-video frame.")
    parser.add_argument("--max-width", type=int, default=1280)
    parser.add_argument("--max-height", type=int, default=850)
    parser.add_argument("--check", action="store_true", help="Validate CSV matches/manifests and decode selected frames; no windows or files written.")
    args = parser.parse_args()
    if args.samples_per_rep < 3 or args.max_width < 640 or args.max_height < 450:
        parser.error("Use at least 3 samples per rep and a display of at least 640x450.")
    if not all(re.fullmatch(r"[A-Za-z0-9_.-]+", code) for code in (args.participant_id, args.annotator_id)):
        parser.error("IDs must contain only letters, digits, dots, underscores and hyphens.")
    single_video = args.single_video or args.video is not None or args.selected_frames is not None
    if args.output_dir is None:
        if single_video:
            args.output_dir = BASE / "person17_manual_annotations"
        else:
            roots = [BASE / "manual_annotations_remaining_videos", BASE / "person17_manual_annotations"]
            existing = [r for r in roots if (r / "batch").is_dir() and
                        (any((r / "batch").rglob("selected_frames_*.csv")) or
                         any((r / "batch").rglob("repetition_review.json")))]
            if len(existing) > 1:
                parser.error("Two earlier batch output folders exist. Specify --output-dir with the one you were using; neither will be changed.")
            args.output_dir = existing[0] if existing else roots[0]
    if not single_video:
        if args.view != "auto":
            parser.error("The batch infers front/side view from each filename; use --view only with --single-video.")
        run_video_directory(args)
        return
    if args.video is None:
        args.video = DEFAULT_VIDEO
    if args.video.stem.casefold() == COMPLETED_VIDEO_NAME.casefold() and not args.selected_frames:
        print("SKIP: protected WhatsApp clip; no annotation changes.")
        for line in audit_whatsapp(args):
            print(line)
        return
    frames = prepare_frames(args)
    if any(clip_name(row['video_name']) == clip_name(COMPLETED_VIDEO_NAME) or
           clip_name(row['source_video_path_private']) == clip_name(COMPLETED_VIDEO_NAME) for row in frames):
        print("SKIP: manifest contains the protected WhatsApp clip; no annotation changes.")
        for line in audit_whatsapp(args):
            print(line)
        return
    annotator = Annotator(frames, args.output_dir, args.annotator_id, args.max_width, args.max_height)
    print(CONVENTION)
    print(f"Matched {frames[0]['video_name']} ({frames[0]['camera_view']}): {len(frames)} frames; {len(frames)*len(JOINTS)} joint slots.")
    print(f"Annotator: {args.annotator_id}. Output: {annotator.output_path}")
    if args.check:
        try:
            for index in range(len(frames)):
                annotator.index = index
                annotator.load_frame()
            print(f"OK: all {len(frames)} frames decoded; {annotator.image_width}x{annotator.image_height} original pixels. No labels written.")
        finally:
            annotator.close()
        return
    manifest = args.output_dir / f"selected_frames_{frames[0]['participant_id']}_{frames[0]['camera_view']}.csv"
    if manifest.exists():
        if read_csv(manifest) != [{k: str(r.get(k, "")) for k in SOURCE_FIELDS} for r in frames]:
            raise ValueError("Existing manifest differs; use --selected-frames with the saved manifest to resume.")
    else:
        write_csv(manifest, SOURCE_FIELDS, frames)
    annotator.run()


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, KeyError, cv2.error) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc

