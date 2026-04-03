# lerobot-doctor

Dataset quality diagnostics for [LeRobot](https://github.com/huggingface/lerobot) v3 datasets.

Catches issues that waste debugging time: corrupted timestamps, dropped frames, frozen actions, clipped values, metadata inconsistencies, video problems, stuck actuators, and more.

Works on local datasets and HuggingFace Hub datasets. No dependency on the lerobot package.

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
# Check a local dataset
lerobot-doctor /path/to/dataset

# Check a HuggingFace dataset
lerobot-doctor lerobot/pusht

# Run specific checks only
lerobot-doctor /path/to/dataset --checks metadata,temporal,actions

# JSON output (for CI/CD integration)
lerobot-doctor /path/to/dataset --json

# Limit episodes checked (for large datasets)
lerobot-doctor /path/to/dataset --max-episodes 10

# Verbose (show PASS details)
lerobot-doctor /path/to/dataset -v
```

## Checks (10 total)

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

Summary: 5 PASS | 5 WARN

Suggested fixes:
  Check sensor connections -- constant readings indicate hardware issues
  Filter episodes shorter than your policy's chunk_size before training
```

## JSON output

Use `--json` for CI integration. Exit code 1 on any FAIL.

```bash
lerobot-doctor /path/to/dataset --json | jq '.overall_severity'
```

## Development

```bash
git clone https://github.com/jashshah999/lerobot-doctor.git
cd lerobot-doctor
pip install -e ".[dev]"
PYTHONPATH=src pytest tests/ -v
```
