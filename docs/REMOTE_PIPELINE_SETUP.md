# Remote AI Referee Pipeline Setup

This guide walks through deploying the fencing referee pipeline on a remote Linux VM and invoking it from another machine (for example, your laptop).

## 1. Repository Layout

Key entry points:
- `AI_Referee.py`: single-phrase pipeline and CLI. Accepts one `.avi` plus matching signal `.txt`, prints a JSON decision, and can emit artifacts (Excel + overlay video).
- `referee_service.py`: FastAPI application exposing the pipeline over HTTP, persisting inputs + artifacts per request.
- `referee_client.py`: Convenience CLI for posting videos/signals to the service and viewing the structured response.
- `training_data/`: sample inputs you can use for smoke testing.

## 2. Server Preparation

1. **System prerequisites**
   - Ubuntu 22.04 (or similar) with Python 3.10+
   - Optional GPU with CUDA drivers if you want hardware acceleration; the code also runs on CPU.

2. **Create a virtual environment (recommended)**
   ```bash
   python3 -m venv ~/ai-referee-venv
   source ~/ai-referee-venv/bin/activate
   python -m pip install --upgrade pip
   ```

3. **Install dependencies**
   ```bash
   pip install fastapi uvicorn[standard] ultralytics opencv-python-headless pandas numpy openpyxl tqdm requests
   ```
   - `ultralytics` will download the YOLO pose model the first time you run the app. To pre-download, run `python - <<'PY'
from ultralytics import YOLO
YOLO('yolov8m-pose.pt')
PY`.

4. **Model weights**
   - Default path is `yolov8m-pose.pt` in the working directory. If you keep it elsewhere, set `REFEREE_YOLO_MODEL=/path/to/weights.pt` before launching the service.

5. **Expose the HTTP port**
   - Decide on a port (default `8000`). Allow inbound traffic through your VM firewall / cloud security group. Example with `ufw`:
     ```bash
     sudo ufw allow 8000/tcp
     ```

## 3. Running the Referee Service

From the project root (inside the virtualenv):
```bash
export REFEREE_HOST=0.0.0.0      # listen on all interfaces
export REFEREE_PORT=8000         # choose any open port
# export REFEREE_OUTPUT_DIR=/data/fencing-artifacts  # optional artifact root
# export GEMINI_API_KEY=...      # optional override, defaults to project key
python referee_service.py
```

The server loads YOLO once on startup and serves:
- `GET /health` – readiness probe
- `POST /analyze` – upload a single `.avi` & `.txt` pair for adjudication. Optional form fields:
  - `include_keypoints=true` to embed per-frame coordinates in the JSON
  - `save_overlay=false` to skip generating the overlay video (defaults to true)

Use `CTRL+C` to stop the service. For production you can wrap it with a process manager (systemd, supervisord, Docker, etc.).

## 4. Optional: Local CLI on the Server

For quick testing without HTTP, run:
```bash
python AI_Referee.py path/to/phrase.avi path/to/phrase.txt \
    --include-keypoints \
    --save-excel \
    --save-overlay \
    --output-dir outputs/phrase01
```
- Output is JSON. Drop `--include-keypoints` for a concise decision.
- `--save-excel` and `--save-overlay` generate artifacts; `--output-dir` controls where they land (created if missing).
- The command loads the YOLO weights on demand; subsequent runs can reuse them by keeping the interpreter alive (e.g., via the service).

## 5. Client Setup (Laptop)

1. Copy `referee_client.py` to your laptop.
2. Ensure Python 3.10+ and install `requests`:
   ```bash
   python3 -m pip install --user requests
   ```
3. Check server health:
```bash
python referee_client.py http://<vm-ip>:8000 --health
```
4. Send a phrase for adjudication (artifacts are stored on the VM under `processed_phrases/` by default):
```bash
python referee_client.py \
    http://<vm-ip>:8000 \
    /path/to/phrase.avi \
    /path/to/phrase.txt \
    --include-keypoints        # optional, large response
    # --no-overlay             # add if you want to skip overlay rendering
```
5. The client prints the JSON response. Capture it into a file if needed:
```bash
python referee_client.py http://<vm-ip>:8000 phrase.avi phrase.txt > decision.json
```

## 6. Response Structure

Successful responses include:
- `status`: `success`
- `winner`: `"left"` or `"right"`
- `reason`: textual explanation of the call
- `natural_language_reason`: Gemini-generated one-sentence summary using fencing terms
- `frames_analyzed`, `normalisation_constant`, `video_angle`
- `phrase`: timings from the electric signal log (start, hit time, pauses)
- `left_pauses` / `right_pauses`: pause/retreat intervals with timing
- `blade_analysis`, `blade_details`, `speed_comparison`: blade/action metrics
- `processing_time_seconds`: core analysis time
- `wall_time_seconds`: end-to-end server wall clock time
- `artifact_dir`: folder containing the persisted artifacts for this phrase
- `artifacts`: paths for Excel, overlay video, stored result JSON, and copies of the uploaded files
- `keypoints`: present only when requested (large payload; per-frame coordinates)

If the TXT file records only one valid hit, the service returns:
```
{
  "status": "skipped",
  "reason": "Only one fencer recorded a valid hit; nothing to adjudicate.",
  "declared_winner": "Right",  # from the electric signal file
  "phrase": { ... }
}
```

## 7. Operational Tips

- The first request after a restart can take longer due to YOLO warm-up.
- For repeated calls, keep the service alive; it shares the preloaded model across requests.
- If you need HTTPS, place a reverse proxy (nginx, Caddy, Traefik) in front of the FastAPI app.
- Monitor resource usage; CPU-only inference is slower, so batch requests accordingly.
- Back up the `.txt` electric signal files with their videos—they are required for a decision.

With these pieces in place, your laptop can upload fencing phrases to the VM, receive structured officiating decisions, and integrate the results into downstream tooling.
