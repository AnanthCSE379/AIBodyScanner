"""Body reconstruction, mesh export, and the existing measurement methods."""

import argparse
import csv
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import trimesh
from scipy.spatial import ConvexHull
from ultralytics import YOLO

# Add 4D-Humans to path so hmr2 package is importable
_HERE = Path(__file__).resolve().parent  # web/
_4DHUMANS = _HERE.parent / "4D-Humans"  # ProjectNomad/4D-Humans/
sys.path.insert(0, str(_4DHUMANS))

from hmr2.configs import CACHE_DIR_4DHUMANS
from hmr2.models import download_models, load_hmr2, DEFAULT_CHECKPOINT
from hmr2.utils import recursive_to
from hmr2.datasets.vitdet_dataset import ViTDetDataset


def calculate_waist_measurements(
    verts,
    joints,
    hip_left_idx=1,
    hip_right_idx=2,
    pelvis_idx=0,
    spine1_idx=9,
    spine2_idx=10,
    spine3_idx=11,
    rshoulder_idx=16,
    lshoulder_idx=17,
):
    """
    Calculate waist using anatomical plane cuts, height scans, and joint ratios.
    Returns: dict with waist_cm, waist_ratio, confidence, method, and notes.
    """
    results = {
        "waist_cm": 0.0,
        "waist_ratio": 0.0,
        "confidence": 0.0,
        "method": "unknown",
        "notes": [],
    }

    if joints is None or len(joints) < 18:
        results["notes"].append("Joints unavailable")
        return results

    hip_left = joints[hip_left_idx]
    hip_right = joints[hip_right_idx]
    hip_level = (hip_left[1] + hip_right[1]) / 2

    # Get spine positions for anatomical plane orientation
    try:
        spine1 = joints[spine1_idx]  # Mid-spine
        spine2 = joints[spine2_idx]  # Upper spine
        spine_vector = spine2 - spine1
        spine_vector = spine_vector / (np.linalg.norm(spine_vector) + 1e-6)
    except:
        spine_vector = np.array([0, 1, 0])

    # Get shoulder positions
    try:
        rshoulder = joints[rshoulder_idx]
        lshoulder = joints[lshoulder_idx]
        shoulder_level = (rshoulder[1] + lshoulder[1]) / 2
    except:
        shoulder_level = joints[spine3_idx][1]

    torso_height = shoulder_level - hip_level
    waist_height = hip_level + 0.38 * torso_height  # Anatomical navel position

    # 1. Anatomical Plane Cut
    try:
        plane_origin = (hip_left + hip_right) / 2
        plane_origin[1] = waist_height

        # Plane normal perpendicular to spine (roughly horizontal but tilted with body)
        # Create a normal by crossing spine vector with frontal direction
        frontal = np.array([0, 0, 1])
        plane_normal = np.cross(spine_vector, frontal)
        plane_normal = plane_normal / (np.linalg.norm(plane_normal) + 1e-6)

        # Get mesh slice at this plane
        mesh = trimesh.Trimesh(vertices=verts, process=False)
        try:
            intersection = trimesh.intersections.mesh_plane(
                mesh, plane_normal=plane_normal, plane_origin=plane_origin
            )

            if intersection is not None and len(intersection) > 0:
                points_3d = []
                for segment in intersection:
                    points_3d.extend(segment)

                if len(points_3d) >= 4:
                    points_3d = np.array(points_3d)
                    hull = ConvexHull(points_3d)
                    boundary = points_3d[hull.vertices]

                    # Calculate perimeter from 3D boundary
                    plane_perimeter = 0.0
                    for i in range(len(boundary)):
                        p1 = boundary[i]
                        p2 = boundary[(i + 1) % len(boundary)]
                        plane_perimeter += np.linalg.norm(p2 - p1)

                    if 0.5 < plane_perimeter < 1.4:
                        results["waist_cm"] = plane_perimeter * 100
                        results["method"] = "anatomical_plane"
                        results["confidence"] = 0.9
                        return results
        except Exception as e:
            results["notes"].append(f"Plane cut failed: {str(e)[:40]}")
    except Exception as e:
        results["notes"].append(f"Plane method error: {str(e)[:40]}")

    # 2. Height Scans around Navel
    try:
        torso_center_xz = (hip_left[[0, 2]] + hip_right[[0, 2]]) / 2

        # Scan tightly around navel
        scan_heights = np.linspace(waist_height - 0.06, waist_height + 0.06, 10)
        measurements_with_confidence = []

        for current_height in scan_heights:
            height_threshold = 0.05  # 5cm slice
            height_diff = np.abs(verts[:, 1] - current_height)
            in_slice = height_diff < height_threshold
            slice_verts = verts[in_slice]

            if len(slice_verts) < 15:
                continue

            # Check for loose clothing: high variance in radial distance suggests baggy clothes
            x_coords = slice_verts[:, 0] - torso_center_xz[0]
            z_coords = slice_verts[:, 2] - torso_center_xz[1]
            radii = np.sqrt(x_coords**2 + z_coords**2)
            radius_variance = np.std(radii)

            # Low variance = tight fit, high variance = loose clothing
            clothing_tightness = max(
                0, 1.0 - (radius_variance / 0.15)
            )  # Normalized [0,1]

            points_2d = np.column_stack([x_coords, z_coords])

            try:
                if len(points_2d) >= 4:
                    hull = ConvexHull(points_2d)
                    boundary = points_2d[hull.vertices]

                    perimeter = sum(
                        np.linalg.norm(boundary[(i + 1) % len(boundary)] - boundary[i])
                        for i in range(len(boundary))
                    )

                    if 0.5 < perimeter < 1.4:
                        # Confidence decreases with distance from expected waist height
                        height_confidence = (
                            1.0 - abs(current_height - waist_height) / 0.15
                        )
                        height_confidence = max(0, height_confidence)

                        # Adjust confidence for clothing tightness
                        combined_confidence = height_confidence * (
                            0.5 + 0.5 * clothing_tightness
                        )

                        measurements_with_confidence.append(
                            {
                                "perimeter": perimeter,
                                "height": current_height,
                                "confidence": combined_confidence,
                                "clothing_score": clothing_tightness,
                            }
                        )
            except:
                continue

        if measurements_with_confidence:
            # Weight by confidence and average
            total_confidence = sum(
                m["confidence"] for m in measurements_with_confidence
            )
            weighted_waist = (
                sum(
                    m["perimeter"] * m["confidence"]
                    for m in measurements_with_confidence
                )
                / total_confidence
            )
            avg_clothing_score = np.mean(
                [m["clothing_score"] for m in measurements_with_confidence]
            )

            results["waist_cm"] = weighted_waist * 100
            results["confidence"] = total_confidence / len(measurements_with_confidence)
            results["method"] = "anatomical_height_scan"

            # Flag if loose clothing detected
            if avg_clothing_score < 0.3:
                results["notes"].append("⚠ Loose clothing detected - low precision")

            return results

    except Exception as e:
        results["notes"].append(f"Height scan error: {str(e)[:40]}")

    # 3. Hip-based Fallback
    try:
        hip_width = np.linalg.norm(hip_left[[0, 2]] - hip_right[[0, 2]])
        hip_circumference = 2 * np.pi * hip_width

        results["waist_cm"] = hip_circumference * 100 * 0.75
        results["confidence"] = 0.4
        results["method"] = "hip_estimation"
        results["notes"].append("Using hip ratio - low confidence")
        return results
    except Exception as e:
        results["notes"].append(f"Hip method error: {str(e)[:40]}")

    return results


def calculate_waist_height_ratio(
    pred_vertices, pred_smpl_params, model, device, person_idx=0, joints_3d=None
):
    """
    Calculate Waist-to-Height ratio using T-pose height and anatomical waist.
    Returns: dict with ratio, waist_cm, height_cm, status, confidence, flags.
    """
    verts = pred_vertices

    if verts.shape[0] == 0:
        return {
            "ratio": 0.0,
            "waist_cm": 0.0,
            "height_cm": 0.0,
            "status": "Invalid",
            "confidence": 0.0,
            "flags": ["No vertices"],
        }

    # Extract Height (T-pose)
    try:
        betas = pred_smpl_params["betas"][person_idx : person_idx + 1]
        batch_pose = pred_smpl_params["body_pose"][person_idx : person_idx + 1]
        batch_orient = pred_smpl_params["global_orient"][person_idx : person_idx + 1]

        # Create T-pose (identity rotation)
        identity_body_pose = (
            torch.eye(3, dtype=batch_pose.dtype, device=batch_pose.device)
            .unsqueeze(0)
            .unsqueeze(0)
            .expand_as(batch_pose)
        )
        identity_global_orient = (
            torch.eye(3, dtype=batch_orient.dtype, device=batch_orient.device)
            .unsqueeze(0)
            .expand_as(batch_orient)
        )

        with torch.no_grad():
            tpose_out = model.smpl(
                betas=betas,
                body_pose=identity_body_pose,
                global_orient=identity_global_orient,
                pose2rot=False,
            )
            tpose_verts = tpose_out.vertices[0].detach().cpu().numpy()

        y_coords = tpose_verts[:, 1]
        y_min = np.min(y_coords)
        y_max = np.max(y_coords)
        height_smpl = y_max - y_min

        if height_smpl < 0.8 or height_smpl > 2.5:
            raise ValueError(f"T-pose height {height_smpl:.4f} out of range [0.8, 2.5]")

        height_cm = height_smpl * 100
        height_method = "T-pose"

    except Exception:
        # Fallback to current pose
        y_coords = verts[:, 1]
        y_min = np.min(y_coords)
        y_max = np.max(y_coords)
        height_smpl = y_max - y_min
        height_cm = height_smpl * 100
        height_method = "current_pose"

    if height_cm <= 0:
        return {
            "ratio": 0.0,
            "waist_cm": 0.0,
            "height_cm": 0.0,
            "status": "Invalid",
            "confidence": 0.0,
            "flags": ["Invalid height"],
        }

    # Extract Waist
    if joints_3d is not None:
        measurement = calculate_waist_measurements(verts, joints_3d)
    else:
        measurement = calculate_waist_measurements(verts, None)

    waist_cm = measurement["waist_cm"]
    confidence = measurement["confidence"]
    method = measurement["method"]
    notes = measurement["notes"]

    # Compute Ratio
    ratio = (waist_cm / 100) / (height_cm / 100)  # Convert cm to m for ratio

    # Classify Health Status (<0.4 Under, <0.5 Healthy, <0.6 Risk, >=0.6 High Risk)
    if ratio < 0.40:
        status = "Underweight"
    elif ratio < 0.50:
        status = "Healthy"
    elif ratio < 0.60:
        status = "Increased Risk"
    else:
        status = "High Risk"

    # Set Confidence Flags
    flags = []

    # Flag low confidence
    if confidence < 0.5:
        flags.append(f"⚠ Low confidence ({confidence:.2f}) - measurement uncertain")
    elif confidence < 0.7:
        flags.append(f"☆ Medium confidence ({confidence:.2f})")

    # Flag if loose clothing detected
    if any("loose clothing" in note.lower() for note in notes):
        flags.append("⚠ Loose clothing - may overestimate waist")

    # Flag unusual measurements
    if ratio < 0.35 or ratio > 0.75:
        flags.append(f"⚠ Unusual ratio {ratio:.3f} - check image quality")

    # Add method info
    flags.append(f"Method: {method}")

    return {
        "ratio": ratio,
        "waist_cm": waist_cm,
        "height_cm": height_cm,
        "status": status,
        "confidence": confidence,
        "flags": flags,
        "height_method": height_method,
    }


def draw_results(img, boxes, measurements_list, person_ids):
    """Draw bounding boxes with Waist-to-Height ratio and confidence on image."""
    img_annotated = img.copy()

    for i, (box, person_id, measurements) in enumerate(
        zip(boxes, person_ids, measurements_list)
    ):
        x1, y1, x2, y2 = map(int, box)

        # Draw bounding box with color based on confidence
        confidence = measurements["confidence"]
        if confidence >= 0.7:
            color = (0, 255, 0)  # Green: high confidence
        elif confidence >= 0.5:
            color = (0, 165, 255)  # Orange: medium confidence
        else:
            color = (0, 0, 255)  # Red: low confidence

        cv2.rectangle(img_annotated, (x1, y1), (x2, y2), color, 2)

        # Prepare text
        ratio = measurements["ratio"]
        status = measurements["status"]
        waist = measurements["waist_cm"]
        conf = measurements["confidence"]

        # Main info line
        text1 = f"Person {person_id}: WHR={ratio:.3f} [{status}]"
        text2 = f"Waist={waist:.0f}cm | Conf={conf:.1%}"

        # Put text above bounding box
        cv2.putText(
            img_annotated, text1, (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
        )
        cv2.putText(
            img_annotated, text2, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1
        )

        # Add warning flag if present
        flags = measurements.get("flags", [])
        if any("⚠" in flag for flag in flags):
            cv2.putText(
                img_annotated,
                "⚠ WARNING",
                (x1, y2 + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2,
            )

    return img_annotated


def load_models(device=None):
    """Load HMR2 model and YOLO detector. Returns (model, model_cfg, detector, device)."""
    if device is None:
        device = (
            torch.device("mps") if torch.mps.is_available() else torch.device("cpu")
        )
    download_models(CACHE_DIR_4DHUMANS)
    model, model_cfg = load_hmr2(DEFAULT_CHECKPOINT)
    model = model.to(device).eval()
    detector = YOLO(str(_HERE.parent / "body_models" / "yolov8x.pt"))
    return model, model_cfg, detector, device


def run_inference(
    img_paths: list, out_folder: str, model, model_cfg, detector, device, batch_size=8
) -> list:
    """
    Process images and return results.

    Returns list of dicts, one per image, each containing:
    {
        'image_name': str,
        'result_image': str (filename of annotated jpg),
        'persons': [
            {
                'person_id': int,
                'waist_cm': float,
                'height_cm': float,
                'ratio': float,
                'status': str,
                'confidence': float,
                'flags': list,
                'obj_file': str (filename of .obj)
            }
        ]
    }
    """
    os.makedirs(out_folder, exist_ok=True)
    all_results = []

    for img_path in img_paths:
        if isinstance(img_path, str):
            img_path = Path(img_path)

        img_cv2 = cv2.imread(str(img_path))
        if img_cv2 is None:
            raise ValueError(f"Could not decode image: {img_path.name}")

        results_yolo = detector(img_cv2, classes=0, conf=0.5)[0]
        boxes = results_yolo.boxes.xyxy.cpu().numpy()

        image_result = {
            "image_name": img_path.name,
            "result_image": None,
            "persons": [],
        }

        if len(boxes) == 0:
            all_results.append(image_result)
            continue

        dataset = ViTDetDataset(model_cfg, img_cv2, boxes)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=0
        )

        measurements_list = []
        person_ids = []

        for batch in dataloader:
            batch = recursive_to(batch, device)
            with torch.no_grad():
                out = model(batch)
            for n in range(batch["img"].shape[0]):
                person_id = int(batch["personid"][n])
                verts = out["pred_vertices"][n].detach().cpu().numpy()

                faces = model.smpl.faces
                if isinstance(faces, torch.Tensor):
                    faces = faces.cpu().numpy()
                faces = np.asarray(faces, dtype=np.int64)
                faces = faces[:, [2, 1, 0]]

                verts = np.asarray(verts, dtype=np.float64)

                mesh = trimesh.Trimesh(
                    vertices=verts, faces=faces, process=False, validate=False
                )

                mesh.fix_normals()

                out_obj_name = f"{img_path.stem}_{person_id}.obj"
                out_path = os.path.join(out_folder, out_obj_name)
                mesh.export(out_path, file_type="obj")

                joints_3d = out["pred_keypoints_3d"][n].detach().cpu().numpy()

                result = calculate_waist_height_ratio(
                    verts, out["pred_smpl_params"], model, device, n, joints_3d
                )

                person_data = {
                    "person_id": int(person_id),
                    "waist_cm": float(result["waist_cm"]),
                    "height_cm": float(result["height_cm"]),
                    "ratio": float(result["ratio"]),
                    "status": str(result["status"]),
                    "confidence": float(result["confidence"]),
                    "flags": [str(f) for f in result["flags"]],
                    "obj_file": out_obj_name,
                }

                image_result["persons"].append(person_data)

                measurements_list.append(
                    {
                        "ratio": result["ratio"],
                        "waist_cm": result["waist_cm"],
                        "height_cm": result["height_cm"],
                        "status": result["status"],
                        "confidence": result["confidence"],
                        "flags": result["flags"],
                    }
                )
                person_ids.append(person_id)

        if len(measurements_list) > 0:
            img_annotated = draw_results(img_cv2, boxes, measurements_list, person_ids)
            result_img_name = f"{img_path.stem}_results.jpg"
            result_img_path = os.path.join(out_folder, result_img_name)
            if not cv2.imwrite(result_img_path, img_annotated):
                raise OSError("Could not save annotated image")
            image_result["result_image"] = result_img_name

        all_results.append(image_result)

    return all_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--img_folder", type=str, default=str(_4DHUMANS / "example_data" / "images")
    )
    parser.add_argument("--out_folder", type=str, default=str(_4DHUMANS / "demo_out"))
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    model, model_cfg, detector, device = load_models()

    img_folder = Path(args.img_folder)
    print(f"Looking for images in: {img_folder.absolute()}")

    img_paths = []
    if img_folder.exists():
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]:
            img_paths.extend(img_folder.glob(ext))

    print(f"Found {len(img_paths)} unique images")

    results = run_inference(
        img_paths, args.out_folder, model, model_cfg, detector, device, args.batch_size
    )

    csv_path = os.path.join(args.out_folder, "waist_height_results.csv")
    with open(csv_path, "w", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(
            [
                "Image",
                "Person_ID",
                "Waist_cm",
                "Height_cm",
                "Waist_Height_Ratio",
                "Status",
                "Confidence",
            ]
        )

        for img_res in results:
            img_name = img_res["image_name"]
            for p in img_res["persons"]:
                csv_writer.writerow(
                    [
                        img_name,
                        p["person_id"],
                        f"{p['waist_cm']:.2f}",
                        f"{p['height_cm']:.2f}",
                        f"{p['ratio']:.4f}",
                        p["status"],
                        f"{p['confidence']:.2f}",
                    ]
                )
                print(f"  Person {p['person_id']}:")
                print(
                    f"    Waist: {p['waist_cm']:.2f} cm | Height: {p['height_cm']:.2f} cm"
                )
                print(f"    Waist-to-Height Ratio: {p['ratio']:.4f} [{p['status']}]")
                print(f"    Confidence: {p['confidence']:.2f}")
                for flag in p["flags"]:
                    print(f"    {flag}")
                print(f"    Saved: {p['obj_file']}")
            if img_res["result_image"]:
                print(f"  Saved annotated image: {img_res['result_image']}")

    print(f"\nResults saved to {csv_path}")
    print("Done!")


if __name__ == "__main__":
    main()
