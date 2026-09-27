"""Check 4: Video Integrity."""

from __future__ import annotations

from pathlib import Path

from lerobot_doctor.dataset_loader import LoadedDataset
from lerobot_doctor.runner import CheckResult, Severity, head, preview


def _probe_video(path: Path, count_frames_if_needed: bool = False) -> dict:
    """Return basic video metadata and whether the first frame decodes.

    PyAV is preferred because it is the package dependency. OpenCV is used as a
    best-effort fallback in development environments where PyAV is unavailable.
    """
    try:
        import av

        container = av.open(str(path))
        stream = container.streams.video[0]
        frame_count = int(stream.frames or 0)
        if frame_count == 0 and count_frames_if_needed:
            container.seek(0)
            frame_count = sum(1 for _ in container.decode(video=0))

        container.seek(0)
        can_decode = False
        for _ in container.decode(video=0):
            can_decode = True
            break

        info = {
            "fps": float(stream.average_rate) if stream.average_rate else None,
            "width": int(stream.width or 0),
            "height": int(stream.height or 0),
            "frames": frame_count,
            "can_decode": can_decode,
            "backend": "PyAV",
        }
        container.close()
        return info
    except ImportError:
        pass

    try:
        import cv2
    except ImportError as e:
        raise RuntimeError("PyAV (av) or OpenCV (cv2) not installed -- skipping video decode checks") from e

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError("could not open video")
    try:
        ok, _ = cap.read()
        return {
            "fps": float(cap.get(cv2.CAP_PROP_FPS) or 0) or None,
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
            "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
            "can_decode": bool(ok),
            "backend": "OpenCV",
        }
    finally:
        cap.release()


def check_videos(dataset: LoadedDataset) -> CheckResult:
    result = CheckResult(name="Video Integrity", severity=Severity.PASS)

    if dataset.info is None:
        result.fail("Cannot check videos: info.json not loaded")
        return result

    # Find video features
    video_features = {
        name: spec for name, spec in dataset.info.features.items()
        if spec.get("dtype") == "video"
    }

    if not video_features:
        result.pass_("No video features declared -- skipping video checks")
        return result

    video_path_template = dataset.info.video_path
    if not video_path_template:
        result.fail("video_path not set in info.json but video features declared")
        return result

    if not dataset.is_local:
        result.warn(
            f"Found {len(video_features)} video feature(s): {list(video_features.keys())} "
            f"-- skipping video decode checks for remote dataset (videos not downloaded)"
        )
        return result

    result.pass_(f"Found {len(video_features)} video feature(s): {list(video_features.keys())}")

    for feat_name, feat_spec in video_features.items():
        _check_video_feature(dataset, feat_name, feat_spec, video_path_template, result)

    return result


def _check_video_feature(
    dataset: LoadedDataset,
    feat_name: str,
    feat_spec: dict,
    video_path_template: str,
    result: CheckResult,
):
    root = dataset.root
    fps = dataset.info.fps if dataset.info else None
    expected_shape = feat_spec.get("shape")  # e.g. [3, 480, 640] or [480, 640, 3]

    # v2 typically has one video file per episode; v3 can store many episodes
    # in the same chunk/file video. Group by resolved path so frame-count checks
    # compare a physical video file with all episodes it contains.
    video_groups: dict[Path, dict] = {}
    missing_videos = []
    decode_errors = []
    fps_mismatches = []
    resolution_mismatches = []
    frame_count_mismatches = []

    for ep_meta in dataset.episodes_meta:
        ep_idx = ep_meta.episode_index
        chunks_size = dataset.info.chunks_size or 1000

        # Get chunk/file indices from episode metadata if available, else compute
        vid_chunk_key = f"videos/{feat_name}/chunk_index"
        vid_file_key = f"videos/{feat_name}/file_index"
        chunk_idx = ep_meta.raw.get(vid_chunk_key, ep_idx // chunks_size)
        # v3 maps episodes to chunk video files via file_index; v2 usually maps
        # to per-episode videos and may not have file_index metadata.
        file_idx = ep_meta.raw.get(vid_file_key, ep_meta.raw.get("episode_index", ep_idx))

        # Build video path from template -- handle different template variable names
        try:
            video_path = video_path_template.format(
                video_key=feat_name,
                episode_chunk=chunk_idx,
                episode_index=file_idx,
                chunk_index=chunk_idx,
                file_index=file_idx,
            )
        except (KeyError, IndexError):
            # Unknown template variables -- skip this feature
            result.warn(f"{feat_name}: Could not resolve video_path template: {video_path_template}")
            return

        full_path = root / video_path

        if not full_path.exists():
            missing_videos.append(ep_idx)
            continue

        group = video_groups.setdefault(full_path, {"episodes": [], "expected_frames": 0})
        group["episodes"].append(ep_idx)
        if ep_meta.length:
            group["expected_frames"] += int(ep_meta.length)
        elif fps:
            from_ts = ep_meta.raw.get(f"videos/{feat_name}/from_timestamp")
            to_ts = ep_meta.raw.get(f"videos/{feat_name}/to_timestamp")
            if from_ts is not None and to_ts is not None:
                group["expected_frames"] += round((to_ts - from_ts) * fps)

    checked = 0
    for full_path, group in sorted(video_groups.items(), key=lambda item: str(item[0])):
        checked += 1
        episodes = group["episodes"]
        first_ep = episodes[0] if episodes else "?"

        try:
            expected_frames = group["expected_frames"]
            probe = _probe_video(full_path, count_frames_if_needed=checked <= 20)

            # Check fps
            if fps and probe["fps"]:
                video_fps = probe["fps"]
                if abs(video_fps - fps) > 1.0:
                    fps_mismatches.append((str(full_path.relative_to(root)), video_fps))

            # Check resolution
            if expected_shape and probe["width"] and probe["height"]:
                # Shape could be [C, H, W] or [H, W, C]
                h, w = probe["height"], probe["width"]
                shape_matches = False
                for perm in [expected_shape, list(reversed(expected_shape))]:
                    if len(perm) >= 2:
                        if (h in perm and w in perm):
                            shape_matches = True
                            break
                if not shape_matches:
                    resolution_mismatches.append((str(full_path.relative_to(root)), h, w, expected_shape))

            # Check frame count vs aggregate episode metadata for this physical file.
            actual_frames = probe["frames"]
            if expected_frames and actual_frames > 0 and abs(actual_frames - expected_frames) > 2:
                frame_count_mismatches.append((
                    str(full_path.relative_to(root)),
                    actual_frames,
                    expected_frames,
                    len(episodes),
                ))

            # Try decoding first frame to check for corruption.
            if not probe["can_decode"]:
                decode_errors.append(first_ep)
        except RuntimeError as e:
            if "not installed" in str(e):
                result.warn(str(e))
                return
            decode_errors.append(first_ep)
        except Exception as e:
            decode_errors.append(first_ep)

        # Only do detailed decode checks on a sample to avoid being too slow.
        if checked >= 20:
            break

    if missing_videos:
        if len(missing_videos) > 10:
            result.fail(
                f"{feat_name}: {len(missing_videos)} video files missing "
                f"(episodes {preview(missing_videos, 5, dataset.no_aggregate)})"
            )
        else:
            result.fail(f"{feat_name}: Video files missing for episodes {missing_videos}")
    else:
        result.pass_(f"{feat_name}: All video files present")

    if decode_errors:
        result.fail(f"{feat_name}: {len(decode_errors)} video(s) failed to decode: episodes {preview(decode_errors, 5, dataset.no_aggregate)}")

    if fps_mismatches:
        shown, hidden = head(fps_mismatches, 3, dataset.no_aggregate)
        for relpath, vfps in shown:
            result.warn(f"{feat_name}: {relpath} video fps={vfps:.1f} != dataset fps={fps}")
        if hidden:
            result.warn(f"{feat_name}: ...and {hidden} more video(s) with fps mismatch")

    if resolution_mismatches:
        shown, hidden = head(resolution_mismatches, 3, dataset.no_aggregate)
        for relpath, h, w, expected in shown:
            result.warn(f"{feat_name}: {relpath} resolution {w}x{h} doesn't match shape {expected}")
        if hidden:
            result.warn(f"{feat_name}: ...and {hidden} more video(s) with resolution mismatch")

    if frame_count_mismatches:
        shown, hidden = head(frame_count_mismatches, 3, dataset.no_aggregate)
        for relpath, actual, expected, n_eps in shown:
            result.warn(
                f"{feat_name}: {relpath} has {actual} frames, expected {expected} "
                f"from {n_eps} episode(s)"
            )
        if hidden:
            result.warn(f"{feat_name}: ...and {hidden} more video(s) with frame count mismatch")
