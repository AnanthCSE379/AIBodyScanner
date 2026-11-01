# BodyScan

A small local app that reconstructs a body mesh from photos with 4D-Humans/HMR2,
shows approximate waist and height measurements, and opens the mesh in Three.js.
FastAPI serves the API and plain HTML/CSS/JavaScript together. No frontend build,
database, account system, or external 3D library CDN is needed.

## Layout

```text
web/
  server.py          # Upload validation, serial job worker, API, static files
  inference.py       # Model loading, mesh export, existing measurement methods
  frontend/
    index.html
    style.css
    app.js           # Uploads, polling, results, CSV
    viewer.js        # Mesh loading, camera, controls, resource cleanup
    vendor/three/    # Three.js 0.170.0 and the two required addons, MIT license
tests/test_server.py  # API checks with stub inference
4D-Humans/           # External research dependency; ignored by Git
body_models/         # Local model weights; ignored by Git
```

Uploaded images and generated artifacts remain in
`4D-Humans/example_data/images/` and `4D-Humans/demo_out/`, so existing files
continue to work. The external model code and licensed weights are not vendored
into this app's source.

## Setup

Use Python 3.9 and the existing `hmr_inference` conda environment if available.
For a fresh checkout, clone the research dependency and apply the Apple Silicon
patch:

```bash
git clone https://github.com/shubham-goel/4D-Humans.git
git -C 4D-Humans apply ../4DHumans_mps_patches.patch
```

For a fresh environment, follow the research project's installation instructions
for its ML dependencies, then install the web app dependencies in that same
environment:

```bash
make install
```

The cleanup has been checked with the existing `hmr_inference` environment;
`requirements.txt` supplies the web/runtime additions, not a complete ML lockfile.

Place the licensed SMPL files in `body_models/smpl/`:

- `basicModel_neutral_lbs_10_207_0_v1.0.0.pkl`
- `SMPL_MALE.pkl`
- `SMPL_FEMALE.pkl`

Obtain these through the [SMPL portal](https://smpl.is.tue.mpg.de/) and follow its
license terms. HMR2 downloads its checkpoint on first startup. The detector uses
`body_models/yolov8x.pt`, downloading it there if it is absent. First startup
requires internet access and may take several minutes. Once weights are cached,
the viewer and inference do not require a CDN connection.

## Run

From the project root:

```bash
conda activate hmr_inference
make dashboard
```

Open [BodyScan](http://127.0.0.1:8000). Wait for the server to finish loading the
models before opening the page. Run **one server process**: the model, queue,
and job history are held in memory. The existing device selection uses MPS on
supported Apple Silicon machines and CPU otherwise.

Select up to 10 JPEG/PNG images, each up to 20 MB and 25 megapixels. Use clear,
full-body photos. You can remove selected images before analysis. The worker
processes one job at a time, with up to four more jobs accepted. A full queue
returns a retryable error. Returning to the upload screen does not cancel an
accepted job.

Results include annotated images, per-person measurements, a CSV download, and
an interactive mesh viewer. Drag to rotate, scroll/pinch to zoom, right-drag to
pan, and use Reset view to frame the mesh. Escape closes the viewer.

## API and files

- `POST /upload`: multipart `files`; optional JSON array `original_names` preserves
  exact display names when multipart header encoding changes them. Returns `job_id`.
- `GET /status/{job_id}`: `queued`, `running`, `done`, or `error`, plus a message.
- `GET /results/{job_id}`: existing per-image `results` and per-person measurements.
  Non-finite values become JSON `null`; the UI shows invalid values as unavailable.
- `GET /files/result-img/{filename}` and `GET /files/obj/{filename}`: output artifacts
  only; filenames may contain spaces/Unicode but cannot be paths or symlinks.
- `DELETE /cleanup/{job_id}`: deletes the job's generated inputs and outputs;
  queued/running jobs return HTTP 409.

The server does not automatically delete completed jobs or photos. Restarting
clears job history but keeps files. Keep this local app bound to localhost.
Upload errors retain the selected files; connection errors during processing
allow retrying status without uploading again.

## Development commands

Activate the Python 3.9 environment first, then use these four targets:

| Command | Purpose |
| --- | --- |
| `make install` | Install app dependencies and the pinned Ruff formatter/linter. |
| `make format` | Format Python code in `web/` and `tests/`. |
| `make lint` | Check Python lint rules and formatting without modifying files. |
| `make dashboard` | Start BodyScan at `http://127.0.0.1:8000`. |

The default `make` command runs `lint`. To select a specific environment without
activating it, use `make lint PYTHON=/path/to/environment/bin/python`.
The research dependency, model files, and vendored Three.js code are excluded from
formatting and linting. These targets cover Python; frontend changes still need
a browser check.

GitHub Actions runs `make lint` on pushes and pull requests. It installs only
`requirements-dev.txt`, so code checks need no model weights or ML downloads.

## Checks

```bash
python -m unittest discover -s tests -v
python -m compileall -q web tests
```

The API tests use stub inference and temporary files, covering upload validation,
filename round-tripping, serialized jobs, queue capacity, model failure recovery,
artifact containment, cleanup, and local viewer modules. A real-model smoke test
also requires the checkpoints, licensed SMPL files, and ML dependencies.

## Measurement limits

The reconstruction and measurement algorithms are unchanged. Single-photo
estimates vary with clothing, pose, and image quality; they are not clinical
measurements. The existing category labels and thresholds are preserved rather
than revalidated by this cleanup. “AI confidence” is the existing measurement
heuristic, not a calibrated probability. Zero, missing, or non-finite measurements
are shown as invalid rather than assigned a category.

## Credits

[4D-Humans](https://github.com/shubham-goel/4D-Humans),
[SMPL](https://smpl.is.tue.mpg.de/),
[Ultralytics](https://github.com/ultralytics/ultralytics), and
[Three.js](https://threejs.org/). Third-party projects and model weights retain
their own licenses.
