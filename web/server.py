"""Local BodyScan API. Run from the project root: python -m uvicorn web.server:app."""

import asyncio
import json
import logging
import math
import queue
import threading
import uuid
import warnings
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "4D-Humans/example_data/images"
OUTPUT_DIR = ROOT / "4D-Humans/demo_out"
FRONTEND_DIR = ROOT / "web/frontend"
MAX_FILES = 10
MAX_FILE_BYTES = 20 * 1024 * 1024
MAX_PIXELS = 25_000_000
MAX_JOBS = 5  # One running job and up to four waiting jobs.
logger = logging.getLogger(__name__)


def load_inference():
    # Import ML dependencies only at startup; API tests can supply a small stub.
    from web.inference import load_models, run_inference

    model, config, detector, device = load_models()

    def infer(paths, output):
        return run_inference(
            paths, str(output), model, config, detector, device, batch_size=8
        )

    return infer


def output_path(directory: Path, filename: str) -> Path:
    """Accept a single artifact name, including spaces and Unicode, never a path."""
    path = directory / filename
    if (
        not filename
        or filename in {".", ".."}
        or "/" in filename
        or "\\" in filename
        or path.is_symlink()
        or path.resolve().parent != directory.resolve()
    ):
        raise HTTPException(400, "Invalid artifact filename")
    return path


def clean_numbers(value):
    """JSON cannot represent NaN or infinity; keep failed measurements explicit."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: clean_numbers(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clean_numbers(item) for item in value]
    return value


def parse_display_names(
    files: list[UploadFile], original_names: Optional[str]
) -> list[str]:
    """Keep exact display names separately from generated storage filenames."""
    if original_names is None:
        return [upload.filename or "" for upload in files]
    try:
        names = json.loads(original_names)
    except (ValueError, TypeError):
        raise HTTPException(400, "Invalid original filenames")
    valid_names = (
        isinstance(names, list)
        and len(names) == len(files)
        and all(isinstance(name, str) and len(name) <= 1024 for name in names)
    )
    if not valid_names:
        raise HTTPException(400, "Invalid original filenames")
    return names


def validate_image(path: Path, suffix: str) -> None:
    """Check the file's actual format, dimensions, and ability to decode."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(path) as image:
                expected_format = "PNG" if suffix == ".png" else "JPEG"
                if (
                    image.format != expected_format
                    or image.width * image.height > MAX_PIXELS
                ):
                    raise ValueError("Unsupported format or image dimensions")
                image.verify()
            # Verification checks structure; loading also catches truncated pixel data.
            with Image.open(path) as image:
                image.load()
    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ):
        raise HTTPException(
            415, "Invalid image. Use a JPEG or PNG up to 25 megapixels."
        )


async def save_upload(upload: UploadFile, directory: Path, job_id: str) -> Path:
    """Stream a size-limited upload to a unique path, then validate its contents."""
    filename = (upload.filename or "").replace("\\", "/").split("/")[-1]
    suffix = Path(filename).suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png"}:
        raise HTTPException(415, "Only JPEG and PNG images are supported")
    path = directory / f"{job_id}_{uuid.uuid4().hex}{suffix}"
    size = 0
    with path.open("wb") as destination:
        while chunk := await upload.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_FILE_BYTES:
                raise HTTPException(413, "Each image must be 20 MB or smaller")
            destination.write(chunk)
    validate_image(path, suffix)
    return path


def create_app(
    inference_loader=load_inference, input_dir=INPUT_DIR, output_dir=OUTPUT_DIR
):
    """Create one local job queue and its routes; tests can inject stub inference."""
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    jobs = {}
    lock = threading.Lock()
    pending = queue.Queue(maxsize=MAX_JOBS)
    slots = threading.BoundedSemaphore(MAX_JOBS)

    def remove_artifacts(job_id):
        # Generated filenames start with this UUID. Also catches partial inference output.
        for directory in (input_dir, output_dir):
            for path in directory.glob(f"{job_id}_*"):
                if path.is_file() and not path.is_symlink():
                    path.unlink(missing_ok=True)

    def worker(infer):
        while True:
            job_id = pending.get()
            if job_id is None:
                pending.task_done()
                return
            try:
                with lock:
                    job = jobs[job_id]
                    job.update(status="running", message="Processing images...")
                results = infer(job["files"], output_dir)
                for result in results:
                    result["image_name"] = job["names"].get(
                        result["image_name"], result["image_name"]
                    )
                results = clean_numbers(results)
                with lock:
                    job.update(
                        results=results, status="done", message="Processing complete"
                    )
            except Exception:
                logger.exception("Job %s failed", job_id)
                try:
                    remove_artifacts(job_id)
                except OSError:
                    logger.exception(
                        "Could not remove partial files for job %s", job_id
                    )
                with lock:
                    jobs[job_id].update(
                        status="error",
                        message="Analysis failed. Check the server log, then try another clear, full-body image.",
                    )
            finally:
                slots.release()
                pending.task_done()

    @asynccontextmanager
    async def lifespan(app):
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        infer = inference_loader()
        thread = threading.Thread(target=worker, args=(infer,), name="bodyscan-worker")
        thread.start()
        try:
            yield
        finally:
            # Finish accepted jobs before stopping the shared model worker.
            await asyncio.to_thread(pending.put, None)
            await asyncio.to_thread(thread.join)

    app = FastAPI(title="BodyScan", lifespan=lifespan)

    @app.middleware("http")
    async def revalidate_frontend(request, call_next):
        response = await call_next(request)
        if request.url.path == "/" or request.url.path.endswith(
            (".html", ".js", ".css")
        ):
            response.headers["Cache-Control"] = "no-cache"
        return response

    @app.post("/upload")
    async def upload_files(
        files: list[UploadFile] = File(...), original_names: Optional[str] = Form(None)
    ):
        reserved = False
        submitted = False
        job_id = str(uuid.uuid4())
        try:
            if not 1 <= len(files) <= MAX_FILES:
                raise HTTPException(400, f"Select 1–{MAX_FILES} images")
            display_names = parse_display_names(files, original_names)
            reserved = slots.acquire(blocking=False)
            if not reserved:
                raise HTTPException(
                    503, "The analysis queue is full. Please try again shortly."
                )
            paths, names = [], {}
            for upload, display_name in zip(files, display_names):
                path = await save_upload(upload, input_dir, job_id)
                paths.append(str(path))
                names[path.name] = display_name
            with lock:
                jobs[job_id] = dict(
                    status="queued",
                    message="Waiting for analysis...",
                    files=paths,
                    names=names,
                    results=[],
                )
            pending.put_nowait(job_id)
            submitted = True
            return {"job_id": job_id}
        finally:
            for upload in files:
                await upload.close()
            if not submitted:
                remove_artifacts(job_id)
                if reserved:
                    slots.release()

    def get_job(job_id):
        if job_id not in jobs:
            raise HTTPException(404, "Job not found. The server may have restarted.")
        return jobs[job_id]

    @app.get("/status/{job_id}")
    async def get_status(job_id: str):
        with lock:
            job = get_job(job_id)
            return {"status": job["status"], "message": job["message"]}

    @app.get("/results/{job_id}")
    async def get_results(job_id: str):
        with lock:
            job = get_job(job_id)
            if job["status"] != "done":
                raise HTTPException(400, "Job is not done yet")
            return {"results": job["results"]}

    def serve_artifact(filename, suffixes, media_type=None):
        path = output_path(output_dir, filename)
        if path.suffix.lower() not in suffixes or not path.is_file():
            raise HTTPException(404, "Artifact not found")
        return FileResponse(path, media_type=media_type)

    @app.get("/files/result-img/{filename:path}")
    async def get_result_image(filename: str):
        return serve_artifact(filename, {".jpg", ".jpeg", ".png"})

    @app.get("/files/obj/{filename:path}")
    async def get_obj_file(filename: str):
        return serve_artifact(filename, {".obj"}, "text/plain")

    @app.delete("/cleanup/{job_id}")
    async def cleanup_job(job_id: str):
        with lock:
            job = get_job(job_id)
            if job["status"] in {"queued", "running"}:
                raise HTTPException(
                    409, "Wait for analysis to finish before deleting this job"
                )
            remove_artifacts(job_id)
            del jobs[job_id]
        return {"status": "success", "message": "Job files deleted"}

    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
    return app


app = create_app()
