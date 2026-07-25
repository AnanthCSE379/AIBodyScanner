import sys
import os
import uuid
import threading
from typing import List, Dict, Any
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import shutil

_BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))   # web/backend/
_WEB_DIR     = os.path.dirname(_BACKEND_DIR)                 # web/
PROJECT_ROOT = os.path.dirname(_WEB_DIR)                     # ProjectNomad/

HUMANS_DIR   = os.path.join(PROJECT_ROOT, "4D-Humans")
INPUT_DIR    = os.path.join(HUMANS_DIR, "example_data", "images")
OUTPUT_DIR   = os.path.join(HUMANS_DIR, "demo_out")
FRONTEND_DIR = os.path.join(_WEB_DIR, "frontend")
WEB_DIR      = _WEB_DIR

# demo_simple lives in web/ (our code, not part of 4D-Humans repo)
sys.path.insert(0, WEB_DIR)
from demo_simple import load_models, run_inference

app = FastAPI(title="BodyScan Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

models = {}
jobs = {}

@app.on_event("startup")
def startup_event():
    print("Loading models...")
    model, model_cfg, detector, device = load_models()
    models["model"] = model
    models["model_cfg"] = model_cfg
    models["detector"] = detector
    models["device"] = device
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(INPUT_DIR, exist_ok=True)
    print("Models loaded successfully.")

def process_job(job_id: str, file_paths: List[str]):
    try:
        jobs[job_id]["status"] = "running"
        jobs[job_id]["message"] = "Processing images..."
        
        results = run_inference(
            img_paths=file_paths,
            out_folder=OUTPUT_DIR,
            model=models["model"],
            model_cfg=models["model_cfg"],
            detector=models["detector"],
            device=models["device"],
            batch_size=8
        )
        
        jobs[job_id]["status"] = "done"
        jobs[job_id]["message"] = "Processing complete"
        jobs[job_id]["results"] = results
    except Exception as e:
        jobs[job_id]["status"] = "error"
        jobs[job_id]["message"] = f"Error during inference: {str(e)}"
        print(f"Job {job_id} failed: {e}")

@app.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    job_id = str(uuid.uuid4())
    os.makedirs(INPUT_DIR, exist_ok=True)
    
    saved_paths = []
    for file in files:
        filename = f"{job_id}_{file.filename}"
        file_path = os.path.join(INPUT_DIR, filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        saved_paths.append(file_path)
    
    jobs[job_id] = {
        "status": "queued",
        "message": "Files uploaded, waiting to process...",
        "files": saved_paths,
        "results": []
    }
    
    # Run in background thread
    thread = threading.Thread(target=process_job, args=(job_id, saved_paths))
    thread.start()
    
    return {"job_id": job_id}

@app.get("/status/{job_id}")
async def get_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    return {
        "status": job["status"],
        "message": job["message"]
    }

@app.get("/results/{job_id}")
async def get_results(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    if job["status"] != "done":
        raise HTTPException(status_code=400, detail="Job is not done yet")
        
    return {"results": job["results"]}

@app.get("/files/result-img/{filename}")
async def get_result_image(filename: str):
    file_path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(file_path)

@app.get("/files/obj/{filename}")
async def get_obj_file(filename: str):
    file_path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="OBJ file not found")
    return FileResponse(file_path, media_type="application/octet-stream")

@app.delete("/cleanup/{job_id}")
async def cleanup_job(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
        
    job = jobs[job_id]
    
    # Delete uploaded images
    for path in job.get("files", []):
        if os.path.exists(path):
            os.remove(path)
            
    # Delete results files if job is done
    if job["status"] == "done":
        for result in job.get("results", []):
            if result.get("result_image"):
                img_path = os.path.join(OUTPUT_DIR, result["result_image"])
                if os.path.exists(img_path):
                    os.remove(img_path)
            
            for person in result.get("persons", []):
                if person.get("obj_file"):
                    obj_path = os.path.join(OUTPUT_DIR, person["obj_file"])
                    if os.path.exists(obj_path):
                        os.remove(obj_path)
                        
    del jobs[job_id]
    return {"status": "success", "message": f"Cleaned up job {job_id}"}

# Mount frontend static files LAST (catch-all)
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
