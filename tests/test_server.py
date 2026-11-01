"""API regression checks with deterministic inference; no model downloads needed."""

import io
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import HTTPException
from fastapi.testclient import TestClient
from PIL import Image

from web.server import create_app, output_path


def image_bytes():
    output = io.BytesIO()
    Image.new("RGB", (8, 8)).save(output, format="PNG")
    return output.getvalue()


class ServerTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.inputs = self.root / "uploads"
        self.outputs = self.root / "results"
        self.release = threading.Event()
        self.release.set()
        self.started = []
        self.inference_fails = False
        self.nonfinite = False
        self.no_person = False
        self.client = TestClient(
            create_app(lambda: self.infer, self.inputs, self.outputs)
        )
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)
        self.addCleanup(self.release.set)

    def infer(self, paths, output):
        self.started.append(paths)
        self.release.wait(5)
        results = []
        for name in paths:
            path = Path(name)
            obj_name = f"{path.stem}_0.obj"
            (output / obj_name).write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
            if self.inference_fails:
                raise ValueError("simulated inference failure")
            results.append(
                dict(
                    image_name=path.name,
                    result_image=None,
                    persons=(
                        []
                        if self.no_person
                        else [
                            dict(
                                person_id=0,
                                waist_cm=80.0,
                                height_cm=180.0,
                                ratio=float("nan") if self.nonfinite else 80 / 180,
                                status="Invalid" if self.nonfinite else "Healthy",
                                confidence=0.5,
                                flags=[],
                                obj_file=obj_name,
                            )
                        ]
                    ),
                )
            )
        return results

    def upload(self, names=("photo.png",)):
        return self.client.post(
            "/upload",
            data={"original_names": json.dumps(names)},
            files=[("files", (name, image_bytes(), "image/png")) for name in names],
        )

    def finish(self, job_id):
        for _ in range(100):
            status = self.client.get(f"/status/{job_id}").json()
            if status["status"] in {"done", "error"}:
                return status
            time.sleep(0.01)
        self.fail("Job did not finish")

    def test_original_names_and_unique_storage(self):
        names = ("same.png", "same.png", 'a ," <script> 測試.png')
        response = self.upload(names)
        self.assertEqual(response.status_code, 200)
        job = response.json()["job_id"]
        self.assertEqual(self.finish(job)["status"], "done")
        results = self.client.get(f"/results/{job}").json()["results"]
        self.assertEqual([result["image_name"] for result in results], list(names))
        self.assertEqual(len(set(self.started[0])), 3)
        for result in results:
            obj = result["persons"][0]["obj_file"]
            response = self.client.get("/files/obj/" + obj)
            self.assertEqual(response.status_code, 200)
            self.assertTrue(response.headers["content-type"].startswith("text/plain"))
        self.assertEqual(self.client.delete(f"/cleanup/{job}").status_code, 200)
        self.assertFalse(list(self.inputs.iterdir()))
        self.assertFalse(list(self.outputs.iterdir()))
        self.assertEqual(self.client.get(f"/status/{job}").status_code, 404)

    def test_invalid_image_removes_partial_uploads(self):
        response = self.client.post(
            "/upload",
            files=[
                ("files", ("good.png", image_bytes(), "image/png")),
                ("files", ("fake.png", b"not an image", "image/png")),
            ],
        )
        self.assertEqual(response.status_code, 415)
        self.assertEqual(list(self.inputs.iterdir()), [])
        self.assertFalse(self.started)

    def test_upload_limits(self):
        self.assertEqual(self.upload(["p.png"] * 11).status_code, 400)
        self.assertEqual(self.upload(["p.gif"]).status_code, 415)
        with patch("web.server.MAX_FILE_BYTES", 8):
            self.assertEqual(self.upload().status_code, 413)
        with patch("web.server.MAX_PIXELS", 4):
            self.assertEqual(self.upload().status_code, 415)
        self.assertEqual(list(self.inputs.iterdir()), [])

    def test_queue_serializes_jobs_and_rejects_overflow(self):
        self.release.clear()
        ids = [self.upload().json()["job_id"] for _ in range(5)]
        self.assertEqual(self.upload().status_code, 503)
        self.assertEqual(len(self.started), 1)
        for job in ids:
            self.assertEqual(self.client.delete(f"/cleanup/{job}").status_code, 409)
            self.assertEqual(self.client.get(f"/results/{job}").status_code, 400)
        self.release.set()
        for job in ids:
            self.assertEqual(self.finish(job)["status"], "done")
        self.assertEqual(len(self.started), 5)

    def test_inference_failure_cleans_outputs_and_worker_recovers(self):
        self.inference_fails = True
        with self.assertLogs("web.server", level="ERROR"):
            job = self.upload().json()["job_id"]
            self.assertEqual(self.finish(job)["status"], "error")
        self.assertFalse(list(self.inputs.iterdir()))
        self.assertFalse(list(self.outputs.iterdir()))
        self.inference_fails = False
        job = self.upload().json()["job_id"]
        self.assertEqual(self.finish(job)["status"], "done")

    def test_nonfinite_measurements_are_json_null(self):
        self.nonfinite = True
        job = self.upload().json()["job_id"]
        self.finish(job)
        person = self.client.get(f"/results/{job}").json()["results"][0]["persons"][0]
        self.assertIsNone(person["ratio"])
        self.assertEqual(person["status"], "Invalid")

    def test_no_people(self):
        self.no_person = True
        job = self.upload().json()["job_id"]
        self.assertEqual(self.finish(job)["status"], "done")
        self.assertEqual(
            self.client.get(f"/results/{job}").json()["results"][0]["persons"], []
        )

    def test_file_routes_are_contained(self):
        secret = self.root / "secret.obj"
        secret.write_text("private")
        (self.outputs / "link.obj").symlink_to(secret)
        for filename in (
            "../secret.obj",
            "subdir/secret.obj",
            "..\\secret.obj",
            "link.obj",
            "/tmp/secret.obj",
        ):
            with self.assertRaises(HTTPException):
                output_path(self.outputs, filename)
        for url in (
            "/files/obj/..%2Fsecret.obj",
            "/files/obj/link.obj",
            "/files/result-img/..%2Fsecret.obj",
        ):
            self.assertEqual(self.client.get(url).status_code, 400)
        self.assertEqual(self.client.get("/files/obj/missing.obj").status_code, 404)
        name = "spaces 測試.obj"
        (self.outputs / name).write_text("v 0 0 0")
        self.assertEqual(self.client.get("/files/obj/" + name).status_code, 200)

    def test_static_frontend_and_local_modules(self):
        for url in (
            "/",
            "/app.js",
            "/viewer.js",
            "/vendor/three/three.module.js",
            "/vendor/three/OrbitControls.js",
            "/vendor/three/OBJLoader.js",
        ):
            self.assertEqual(self.client.get(url).status_code, 200, url)
        self.assertNotIn("unpkg.com", self.client.get("/").text)
        self.assertEqual(self.client.get("/status/unknown").status_code, 404)


if __name__ == "__main__":
    unittest.main()
