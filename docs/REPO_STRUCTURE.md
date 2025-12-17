# Repository Structure and Script Usage

## Top-Level Layout

- `src/` – core Python packages.
  - `src/ml/` – data loading, feature extraction, and model utilities (`data_loader.py`, `feature_extraction.py`, etc.).
  - `src/referee/` – end-to-end analysis pipeline, including tracking, blade-contact heuristics, and the ensemble models.
  - `src/utils/` – shared helpers (YOLO integration, YouTube utilities).
- `scripts/` – runnable utilities and CLI tools (documented below).  Legacy scripts depending on older APIs live in `scripts/legacy/` and are kept for reference only.
- `data/`
  - `data/training_data/` – original phrase folders containing TXT, video, Excel, and JSON artifacts.
  - `data/blade_touch_data/` – curated subset produced by `extract_blade_touch_data.py` where the final blade contact occurs within one second of the hit.
- `results/`
  - `results/blade_touch_experiments/` – archived model artifacts from the feature-engineering work.
  - `results/blade_touch_verification/` – populated by `run_blade_touch_verification.py` (separate `correct/` and `mismatch/` folders plus `verification_results.csv`).
  - `results/correct_results/`, `results/mismatched_results/` – populated by `validate_all_results.py` when re-checking the original training set.
- `docs/` – project documentation (this file plus any existing design notes).
- `models/` – place YOLO weights (e.g., `models/yolo11x-pose.pt`) for the heavy re-processing scripts.

## Scripts

| Script | Description & Key Flags |
| --- | --- |
| `blade_touch_referee.py` | Scans `data/blade_touch_data`, builds the logistic feature table, trains the logistic model, and writes predictions + model artifacts. Run: `python3 scripts/blade_touch_referee.py`. |
| `debug_referee.py` | Given a phrase folder name, reruns pause detection + the logistic blade-touch decision and prints a JSON summary. Usage: `python3 scripts/debug_referee.py <folder_name>` with folders relative to `data/training_data`. |
| `evaluate_debug_referee.py` | Batch-evaluates `debug_referee` across `data/blade_touch_data` and writes `results/debug_referee_eval.csv`. Run: `python3 scripts/evaluate_debug_referee.py`. |
| `run_blade_touch_verification.py` | Replays the blade-touch phrases, splits them into `results/blade_touch_verification/{correct,mismatch}` directories, and records accuracy. Run: `python3 scripts/run_blade_touch_verification.py`. |
| `extract_blade_touch_data.py` | Copies phrases whose last blade contact is within N seconds of the hit into `data/blade_touch_data`. Usage: `python3 scripts/extract_blade_touch_data.py [training_dir] [output_dir] [--threshold SECONDS] [--dry-run]`. |
| `debug_features.py` | Prints the engineered feature dictionary for a sample phrase (skips folders without winners). Usage: `python3 scripts/debug_features.py [--data-dir DIR] [--folder NAME]`. |
| `inspect_excel.py` | Summaries the sheets/columns within a keypoint Excel file. Usage: `python3 scripts/inspect_excel.py [path/to/file.xlsx]` (auto-discovers the first file under `data/training_data` if unspecified). |
| `run_ai_judged_export.py` | Trains the ensemble model on the feature table built from `data/training_data` and copies videos into `results/AI_Judged_{Correct,Mismatched}`. Flags: `--root DIR`, `--output-correct DIR`, `--output-mismatch DIR`, `--max-phrases N`, `--dry-run`. Requires enough labeled phrases per class; otherwise exits early with a notice (useful for quick smoke tests). |
| `validate_all_results.py` | Re-runs the `debug_referee` logic across `data/training_data`, updating each `analysis_result.json` and copying phrases into `results/{correct,mismatched}_results`. Flags: `--root DIR`, `--correct-dir DIR`, `--mismatch-dir DIR`, `--limit N`. |
| `full_reprocess.py` | Performs fisheye correction, reruns YOLO tracking, produces Excel/overlay files, and copies mismatches. New flags: `--training-dir`, `--mismatch-dir`, `--model-path`, `--limit`. Set `--limit 0` (or point to non-existent weights) for a dependency check without work. Requires `models/yolo11x-pose.pt`. |
| `reprocess_all.py` | Similar to `full_reprocess.py` but skips fisheye correction. Supports the same flags/short-circuit behavior. |
| `run_blade_touch_verification.py` | (See above) |
| `evaluate_debug_referee.py` | (See above) |
| `run_ai_judged_export.py` | (See above) |

Legacy helpers (`scripts/legacy/AI_Referee_old.py`, `scripts/legacy/ML_Referee.py`) depend on the obsolete `AI_Referee` module and are preserved only for reference; use the modern `src/referee` pipeline instead.

## Key Data & Result Folders

- `data/training_data/<timestamp_phrase>/` – raw artifact folders.
- `data/blade_touch_data/<timestamp_phrase>/` – curated subset produced by `extract_blade_touch_data.py`.
- `results/blade_touch_verification/{correct,mismatch}/` – phrase copies organized by the logistic pipeline outcome.
- `results/blade_touch_experiments/` – archived scripts/model outputs for reproducibility.
- `results/correct_results/` & `results/mismatched_results/` – populated by `validate_all_results.py` when scanning the full training set.

Use this document as the starting point when navigating the repository or invoking the various processing scripts.
