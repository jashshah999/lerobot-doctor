# lerobot-doctor

Dataset quality diagnostics for [LeRobot](https://github.com/huggingface/lerobot) v2/v3 datasets.

Catches issues that waste debugging time: corrupted timestamps, dropped frames, frozen actions, clipped values, metadata inconsistencies, video problems, stuck actuators, and more.

Works on local datasets and HuggingFace Hub datasets. No dependency on the lerobot package.

**Live now** on the [LeRobot Dataset Visualizer](https://huggingface.co/spaces/lerobot/visualize_dataset) as the "Doctor" tab, and as a standalone [HF Space](https://huggingface.co/spaces/jashshah999/lerobot-doctor).

## Install

```bash
pip install lerobot-doctor
```

Or from source:

```bash
git clone https://github.com/jashshah999/lerobot-doctor.git
cd lerobot-doctor
pip install .
```

## Usage

```bash
# Check a local dataset directory
lerobot-doctor /path/to/dataset

# Check a local dataset .zip archive (format version is auto-detected)
lerobot-doctor /path/to/dataset.zip

# Check a HuggingFace dataset
lerobot-doctor lerobot/pusht

# Run specific checks only
lerobot-doctor /path/to/dataset --checks metadata,temporal,actions

# JSON output (for CI/CD integration)
lerobot-doctor /path/to/dataset --json

# Markdown report (paste into PRs or dataset cards)
lerobot-doctor /path/to/dataset --markdown report.md

# Limit episodes checked (recommended for huge HF datasets like lerobot/droid_1.0.1)
lerobot-doctor lerobot/droid_1.0.1 --max-episodes 10

# Verbose (show PASS details)
lerobot-doctor /path/to/dataset -v

# CI mode: JSON output, exits 1 on failures
lerobot-doctor /path/to/dataset --ci

# CI mode: exits 1 on warnings too
lerobot-doctor /path/to/dataset --ci --fail-on=warn
```

## Checks (12 total)

| Check | What it catches |
|-------|----------------|
| **metadata** | Missing/invalid info.json, wrong episode/frame counts, missing data files, tasks.parquet issues |
| **temporal** | Non-monotonic timestamps, dropped frames, inconsistent fps, broken frame/episode indices |
| **actions** | NaN/Inf values, clipped actions, frozen (stuck) actions, sudden action jumps |
| **videos** | Missing video files, decode errors, fps/resolution mismatches, frame count mismatches |
| **statistics** | NaN/Inf in observations, zero-variance features, extreme outliers, stats.json drift |
| **episodes** | Short/empty episodes, length distribution, policy window compatibility (ACT/Diffusion), metadata-data length mismatches, task imbalance |
| **consistency** | Cross-episode feature schema changes (missing columns, dtype/shape mismatches), within-episode shape inconsistencies |
| **training** | Policy compatibility (ACT/Diffusion/VLA), normalization readiness (zero-std dims), action space sanity, delta_timestamps compatibility |
| **anomalies** | Stuck actuators (>80% static), near-duplicate episodes, distribution shift across dataset, broken sensors (constant observations) |
| **portability** | Absolute paths, symlinks, large files, HF Hub compatibility, non-standard files |
| **per_episode** | Per-episode drilldown: flags specific bad episodes with reasons (short, frozen, NaN, timestamp gaps, action jumps) |
| **kinematics** | Actions that violate a robot's *real* physical joint limits (position, and implied velocity), from a URDF -- distinct from `actions`/`statistics`, which only check against the dataset's own internal min/max/variance |

### kinematics: checking against real robot limits, not just dataset statistics

Every other check validates a dataset against itself. `kinematics` is the odd one out: it
validates actions against an external ground truth -- the robot's actual joint limits --
so it can catch a command that's statistically unremarkable (no clipping, no jumps, no
NaNs) but physically unreachable.

```bash
# Uses a small built-in registry, keyed by the dataset's robot_type (currently: so100, so101)
lerobot-doctor lerobot/svla_so101_pickplace --checks kinematics

# Or supply any robot's URDF directly
lerobot-doctor /path/to/dataset --checks kinematics --urdf /path/to/robot.urdf
```

**Known limitation:** some LeRobot robot configs (several SO-100/SO-101 setups included)
store actions as a normalized percentage (roughly `[-100, 100]`) rather than the URDF's
radians. Checking those directly would flag nearly every frame as a "violation" -- not a
real kinematic problem, just a unit mismatch. The check guards against this: if most
values are out of bounds *and* their typical magnitude is more than 10x the limit's scale,
it's reported as a likely unit mismatch and skipped, not failed. This guard isn't perfect --
validated against `lerobot/svla_so101_pickplace`, it correctly caught the mismatch on 5 of
6 joints; the 6th (`gripper`) has a URDF range so small (~1.9 rad wide) that a percent-scale
value doesn't clear the 10x bar, so it's still reported as a false "violation" there. There's
no reliable general fix without a units field in the dataset itself, which LeRobot doesn't
provide today -- flagging this here for visibility rather than hiding it.

## Exit codes

- `0`: All checks PASS or WARN
- `1`: At least one check FAIL

## Example output

```
lerobot-doctor v0.1.0 -- Dataset Quality Report
Dataset: lerobot/pusht (v3.0)
Episodes: 206 | Frames: 25,650 | FPS: 10

[PASS] Metadata & Format Compliance
[PASS] Temporal Consistency
[WARN] Action Quality
  - action: Episode 2 has 1 sudden large action jumps (>5 std)
  - action: Episode 3 has 2 sudden large action jumps (>5 std)
[WARN] Video Integrity
  - Skipping video decode checks for remote dataset
[WARN] Data Distribution
  - next.success: zero variance (constant value 0.0000)
[WARN] Episode Health
  - 2/10 episodes shorter than chunk_size=100 (used by ACT/Diffusion policies)
[PASS] Feature Consistency
[PASS] Training Readiness
[WARN] Anomaly Detection
  - next.success: ALL 1 dimensions constant across ALL episodes
[WARN] Per-Episode Drilldown
  - Episode 2: 1 sudden action jumps
  - Episode 3: 2 sudden action jumps
[PASS] Kinematic Feasibility
  - Skipped: no --urdf given and robot_type='pusht' is not in the built-in registry

Summary: 6 PASS | 6 WARN

Suggested fixes:
  Check sensor connections -- constant readings indicate hardware issues
  Filter episodes shorter than your policy's chunk_size before training
```

## CI / GitHub Actions

Use `--ci` for pipeline integration. Outputs JSON to stdout, one-line summary to stderr.

```bash
# Fail pipeline on any FAIL
lerobot-doctor lerobot/pusht --ci

# Fail pipeline on WARNs too (stricter)
lerobot-doctor lerobot/pusht --ci --fail-on=warn
```

Example GitHub Actions step:

```yaml
- name: Dataset quality gate
  run: lerobot-doctor my-org/my-dataset --ci --fail-on=warn
```

## JSON output

Use `--json` for JSON output without CI exit-code behavior.

```bash
lerobot-doctor /path/to/dataset --json | jq '.overall_severity'
```

## Huge datasets

For very large datasets (e.g. `lerobot/droid_1.0.1` at ~28M frames across 156 data parquets + videos), always pass `--max-episodes N` with a small N (10-100). Running without it attempts a full download, which:

- On the hosted [HF Space](https://huggingface.co/spaces/jashshah999/lerobot-doctor): **will fail** -- the Space has ~50GB ephemeral disk. The Space also blocks "all episodes" on datasets with >1M frames.
- Locally: works if you have the bandwidth and disk, but is slow.

When `--max-episodes` is set on a HF dataset, lerobot-doctor:

1. Fetches small meta files (info.json, tasks.parquet, stats.json).
2. Downloads episodes meta parquets one at a time until the first N episodes are covered.
3. Resolves `data/chunk_index` + `data/file_index` to pull only the data parquets that contain those N episodes.

Checks that rely on the full dataset (e.g. `total_episodes` count in metadata) are automatically skipped in partial mode instead of flagging false-positive failures.

**Future work:** sampled full-dataset scans (random subset across chunks), video sampling without full download.

## Fix (auto-repair)

```bash
lerobot-doctor fix /path/to/dataset              # fix all issues (creates backup)
lerobot-doctor fix /path/to/dataset --dry-run    # preview fixes
lerobot-doctor fix /path/to/dataset --fixes timestamps,metadata
```

Fixes: broken indices, timestamp drift, NaN values, metadata mismatches.

## Trim (remove idle frames)

```bash
lerobot-doctor trim /path/to/dataset             # remove start/end idle frames
lerobot-doctor trim /path/to/dataset --dry-run   # preview
```

Fixes the "robot does nothing at inference" problem caused by static training data.

## Score (find bad episodes)

```bash
lerobot-doctor score /path/to/dataset            # rank episodes by quality
lerobot-doctor score /path/to/dataset --json     # JSON for automation
```

Scores: smoothness, coverage, consistency, length. Recommends which episodes to drop.

## Gate (pre-training check)

```bash
lerobot-doctor gate /path/to/dataset --policy act        # ACT compatibility
lerobot-doctor gate /path/to/dataset --policy diffusion  # Diffusion Policy
lerobot-doctor gate /path/to/dataset --policy smolvla    # SmolVLA
lerobot-doctor gate /path/to/dataset --policy pi0        # Pi0
```

Exit code 1 if dataset is incompatible. Catches: wrong dims, short episodes, NaN normalization.

## Merge Check

```bash
lerobot-doctor merge-check ./dataset1 ./dataset2         # pre-merge compatibility
lerobot-doctor merge-check ./merged_dataset --post-merge # post-merge validation
```

## Development

```bash
git clone https://github.com/jashshah999/lerobot-doctor.git
cd lerobot-doctor
pip install -e ".[dev]"
PYTHONPATH=src pytest tests/ -v
```
