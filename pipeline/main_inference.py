# pipeline/main_inference.py
import os
import pandas as pd
import argparse
import traceback
import time
from pathlib import Path

from fetch_image import fetch_satellite_image
from model_inference import load_model, predict, get_default_device
from postprocess import (
    format_output,
    encode_mask_as_polygon
)
from utils.io_utils import save_json, ensure_dir

from utils.buffer_utils import (
    compute_gsd_m_per_px,
    compute_buffer_radius_px,
    create_circular_buffer,
    compute_overlap_area
)

import numpy as np
import torch


def run_inference(coords_file, output_dir, overlay_dir, api_key, benchmark_runs: int = 0):
   
    ensure_dir(output_dir)
    ensure_dir(overlay_dir)

    df = pd.read_excel(coords_file)

    # ----- Device & Model -----
    device = get_default_device()
    print(f"[INFO] Using device: {device}")
    model = load_model(device=device)

    results = []

    # Optional benchmarking variables
    times = []

    for i, (_, row) in enumerate(df.iterrows()):
        sid = row["sample_id"]
        lat, lon = float(row["lat"]), float(row["lon"])

        print(f"\n[PROCESSING] {sid} @ ({lat}, {lon})")

        try:
            # ----- Fetch satellite image (API key passed in) -----
            img = fetch_satellite_image(lat, lon, api_key=api_key)
            img_arr = np.array(img)

            H, W = img_arr.shape[:2]

            # ----- Dynamic GSD -----
            gsd = compute_gsd_m_per_px(lat)

            # ----- Compute buffers -----
            r1200 = compute_buffer_radius_px(1200, gsd)
            r2400 = compute_buffer_radius_px(2400, gsd)

            buf1200 = create_circular_buffer((H, W), r1200)
            buf2400 = create_circular_buffer((H, W), r2400)

            # ----- Model inference -----
            start_t = time.time()
            mask_img, conf = predict(model, img, device=device)
            elapsed = time.time() - start_t

            # record elapsed if benchmarking
            if benchmark_runs and len(times) < benchmark_runs:
                times.append(elapsed)

            mask_arr = np.array(mask_img)

            # ----- PV inside 1200 sq ft -----
            area_1200 = compute_overlap_area(mask_arr, buf1200, gsd)

            if area_1200 > 1.0:
                has_solar = True
                pv_area_m2 = area_1200
                buffer_used_sqft = 1200
            else:
                # fallback: rooftop presence inside 2400 sq ft
                area_2400 = compute_overlap_area(mask_arr, buf2400, gsd)
                has_solar = False
                pv_area_m2 = area_2400
                buffer_used_sqft = 2400

            # ----- Polygon for explainability -----
            polygon_b64 = encode_mask_as_polygon(mask_img)

            # ----- Record -----
            rec = format_output(
                sample_id=sid,
                lat=lat,
                lon=lon,
                has_solar=has_solar,
                confidence=conf,
                pv_area=pv_area_m2,
                buffer_radius_sqft=buffer_used_sqft,
                bbox_or_mask=polygon_b64,
            )

            results.append(rec)

            # Save overlay
            try:
                mask_img.save(os.path.join(overlay_dir, f"{sid}.png"))
            except Exception:
                pass

            print(f"[OK] Completed {sid}  (inference {elapsed:.3f}s)")

            # If we collected enough benchmark samples, stop collecting
            if benchmark_runs and len(times) >= benchmark_runs:
                # If the user only wanted benchmarking, we will still process all samples,
                # but we've already captured enough timing samples.
                pass

        except Exception:
            print(f"[ERROR] Sample {sid} failed.")
            traceback.print_exc()

    out_path = os.path.join(output_dir, "predictions.json")
    save_json(results, out_path)
    print(f"\n[SAVED] {out_path}")

    # Report benchmark summary if requested
    if benchmark_runs and len(times) > 0:
        avg = sum(times) / len(times)
        print(f"\n[BENCHMARK] Ran {len(times)} timed inferences. Avg inference time: {avg:.3f}s (device={device})")

    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PV detection inference on coordinates XLSX")
    parser.add_argument("coords_file", type=str, help="Excel file with columns: sample_id, lat, lon")
    parser.add_argument("--output_dir", type=str, default="out", help="Directory to save predictions.json")
    parser.add_argument("--overlay_dir", type=str, default="overlays", help="Directory to save overlay PNGs")
    parser.add_argument("--api_key", type=str, required=True, help="Google Static Maps API key")
    parser.add_argument("--benchmark", type=int, default=0, help="Number of inferences to time (for benchmarking)")

    args = parser.parse_args()

    run_inference(args.coords_file, args.output_dir, args.overlay_dir, api_key=args.api_key, benchmark_runs=args.benchmark)
