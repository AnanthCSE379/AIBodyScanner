# ProjectNomad — Body Composition Estimator

An AI-powered web app that estimates body composition metrics (waist-to-height ratio) from a single photo. Upload an image of a person, and the system outputs their estimated waist circumference, height, and health classification.

---

## What it does

1. You upload one or more images through a web interface.
2. YOLOv8 detects every person in the image.
3. For each detected person, the 4D-Humans model (HMR2) reconstructs a 3D body mesh from the 2D photo.
4. The waist and height are measured directly from that 3D mesh.
5. A waist-to-height ratio (WHR) is calculated and classified into a health category.
6. You get back the annotated image, the measurements, and (optionally) the 3D mesh to look at.

---

## How it works under the hood

### 4D-Humans / HMR2

4D-Humans is a research model from the University of Pennsylvania that estimates the 3D pose and shape of a human body from a single RGB image. It outputs a SMPL body mesh — a standard parametric human body model with ~6890 vertices.

The key insight: instead of measuring waist size from the photo directly (which is basically impossible accurately), we reconstruct the 3D body, then slice the mesh at the anatomically correct waist level (around the navel), and measure the cross-section. This gives a real centimeter estimate, not a pixel count.

Internally it uses:
- **ViT-based backbone** (Vision Transformer) to encode image features
- **SMPL body model** to output the body mesh
- **YOLOv8x** for person detection before the body reconstruction step

### Waist measurement approach

The waist is estimated using a combination of three methods:
1. A horizontal plane cut at the navel height (midpoint between hip joints and shoulder joints)
2. A convex hull of the cross-section to get the perimeter
3. A joint-based width ratio as a fallback sanity check

The final waist number is a weighted blend of these, with a confidence score that tells you how reliable the estimate is.

### Health classification (WHR thresholds)

| WHR | Category |
|-----|----------|
| < 0.40 | Underweight |
| 0.40 – 0.49 | Healthy |
| 0.50 – 0.59 | Increased Risk |
| ≥ 0.60 | High Risk |

These are based on established clinical guidelines.

---

## Tech stack

| Layer | What it uses |
|-------|-------------|
| Body reconstruction | 4D-Humans (HMR2) + SMPL |
| Person detection | YOLOv8x (Ultralytics) |
| Backend API | FastAPI + Uvicorn |
| Frontend | Vanilla HTML, CSS, JavaScript |
| 3D mesh viewing | Three.js + OBJLoader |
| Python environment | Python 3.9 (Required) |

---

## Project structure

```
ProjectNomad/
├── 4DHumans_mps_patches.patch  # Patch to make 4D-Humans run on Apple Silicon
├── 4D-Humans/                  # Clone this separately — not included in repo
│   ├── example_data/images/    # Input images go here temporarily
│   ├── demo_out/               # Output: annotated images, .obj files, CSV
│   └── hmr2/                   # 4D-Humans model code
│
└── web/
    ├── demo_simple.py          # Core inference logic (load_models, run_inference)
    ├── backend/
    │   └── main.py             # FastAPI server, job queue, file serving
    └── frontend/
        ├── index.html
        ├── style.css
        └── app.js
```




## How to run it

Complete setup in order:

**1. Clone this repo**
```bash
git clone https://github.com/yourusername/ProjectNomad.git
cd ProjectNomad
```

**2. Clone 4D-Humans and apply the MPS patch**
```bash
git clone https://github.com/shubham-goel/4D-Humans.git
cd 4D-Humans
git apply ../4DHumans_mps_patches.patch
cd ..
```

**3. Install Python dependencies (Python 3.9 required)**
```bash
pip install torch torchvision torchaudio
pip install trimesh scipy numpy opencv-python
pip install ultralytics
pip install fastapi uvicorn python-multipart
```

**4. Download SMPL model files manually**

Place these three files in `body_models/smpl/`:
- `basicModel_neutral_lbs_10_207_0_v1.0.0.pkl`
- `SMPL_MALE.pkl`
- `SMPL_FEMALE.pkl`

Download from the [HMR 2.0 / 4D-Humans website](https://shubham-goel.github.io/4dhumans/) or the [SMPL portal](https://smpl.is.tue.mpg.de/).

*(The YOLOv8 weights and HMR2 model checkpoint download automatically on first run.)*

**5. Start the server**
```bash
cd web/backend
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

> **Important:** Run this with the Python environment that has all the ML packages installed. If you're using conda: `conda run -n <your_env> --no-capture-output python -m uvicorn main:app --host 0.0.0.0 --port 8000`

Then open **http://localhost:8000** in your browser.

First startup takes 1–2 minutes — the HMR2 model checkpoint and YOLO weights need to load into memory. After that it stays warm until you stop the server.

---

## Limitations

**Accuracy is rough, not clinical-grade.**
The waist estimate can be off by 5–15 cm depending on clothing, pose, and image quality. This is expected — we're estimating a 3D body from a single 2D photo, which is an inherently ambiguous problem. The model makes assumptions about body shape that may not match reality.

**It needs a clear, full-body photo.**
The body reconstruction works best when the whole person is visible — head to toe. Cropped images, extreme angles, or people sitting down will give bad or failed results. The person should ideally be facing the camera roughly straight-on.

**One person works better than many.**
While it can handle multiple people in a frame, accuracy drops in crowded images because the detection boxes can overlap and the mesh reconstruction gets confused.

**Clothing throws it off.**
The model estimates the underlying body shape, but it was trained mostly on images without heavy winter clothing. Thick jackets, baggy clothes, etc. will cause the mesh to bloat and overestimate waist size.

**It runs on MPS (Apple Silicon) right now.**
The inference runs on the Mac GPU via PyTorch MPS. This works but is slower than CUDA and occasionally triggers warnings about unsupported operations that fall back to CPU. On a machine with an NVIDIA GPU, switching to CUDA would be significantly faster — this is not yet configured.

---

## Work in progress / known issues

**3D mesh viewer.**
The "View 3D Model" button in the UI currently doesn't load the model properly. The OBJ file is generated correctly and saved to `demo_out/`, but there's an issue with how it gets served and loaded in the browser.

**No CUDA support configured.**
The backend auto-selects MPS on Apple Silicon. For a machine with an NVIDIA GPU, you'd need to change the device selection logic in `demo_simple.py`. This is straightforward but not yet done.

**The app serves one job at a time.**
App to be scaled to concurrently handle multiple jobs soon.

**Images are named after the job ID + original filename.**
If your original file has spaces in the name (e.g. `arun j.png`), that can cause URL encoding issues in some browsers when trying to view the results. Renaming your files to use underscores before uploading avoids this.

**CSV download.**
There's a "Download CSV" button in the results view that isn't wired up yet.

---

## What's next

- Fix the 3D OBJ viewer
- Wire up the CSV download
- CUDA support for non-Mac machines  
- Better error messages when a person isn't detected
- Batch processing improvements and a proper job queue
- Height calibration — right now height is estimated from the mesh which can be off; using a reference object in the frame would improve accuracy a lot

---

## Credits

- [4D-Humans (HMR2)](https://github.com/shubham-goel/4D-Humans) by Goel et al., UPenn — the body reconstruction model
- [SMPL](https://smpl.is.tue.mpg.de/) — the parametric body model
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) — person detection