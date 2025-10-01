#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gptnew3_MATCH_GEMINI_PARITY_OPENAI_CLEAN.py

Paritas dengan skrip Gemini:
- Per-gambar: jika ADA tanda varises (meski halus) → label="Varices" (grade=1 bila subtle). Tidak perlu ≥2 tanda.
- Ringkasan per-folder: hanya frame quality=="OK"; threshold proporsi varices 0.03; minimal OK-frames 5; grade=modus (fallback median).
- Chunking & error handling: MAX_IMAGES_PER_CHUNK_STEPS=[30,20,10], MIN_CHUNK_IMAGES=5, binary shrink, delay antar call.
- Output JSON Lines per gambar; evidence ≤3 kata.
- Excel: Ringkasan, Detail Batch, Per-Image, Batch Raw.
- Estimasi biaya pra-run & live.

Kebutuhan:
  pip install openai>=1.40 pillow pandas xlsxwriter
"""

import os, sys, time, json, argparse, statistics as stats
from io import BytesIO
from typing import List, Dict, Tuple
import pandas as pd

OPENAI_MODEL               = "gpt-4o-mini"
MAX_REQUEST_BYTES          = 17 * 1024 * 1024
MAX_IMAGES_PER_CHUNK_STEPS = [30, 20, 10]
MIN_CHUNK_IMAGES           = 5
CALL_DELAY_SECONDS         = 0.7
ERROR_LOWERING_THRESHOLD   = 3
SINGLE_REQUEST_IF_POSSIBLE = True

VARICES_PROP_THRESHOLD     = 0.03
MIN_OK_QUALITY_FRAMES      = 5
MIN_VAR_FRAMES_REQUIRED    = 0
MIN_CONF_COUNT             = 0

TEMPERATURE                = 0.05
EXPECTED_OUTPUT_TOKENS     = 50
MAX_OUTPUT_CAP             = 2048

TARGET_KB                  = 24
SAMPLE_EVERY               = 10

PRICE_INPUT_PER1K          = 0.00015
PRICE_OUTPUT_PER1K         = 0.0006
BUDGET_USD                 = 4.0
ESTIMATE_SAMPLES           = 200

try:
    from PIL import Image
    PIL_OK = True
except Exception:
    PIL_OK = False

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

PROMPT_CLASSIFY = """You are a pediatric gastroenterology expert.
You will receive N endoscopy images in order. First, read the index-to-filename mapping.
Then, for EACH image 1..N, output EXACTLY one compact JSON on its own line (JSON Lines) and nothing else.
Schema: {"image":"<filename>","label":"Normal|Varices","grade":0|1|2|3,"confidence":0-100,"quality":"OK|Blurred|OutOfEsophagus","evidence":["<=3 words","very short","hints"]}
Rules:
- If ANY variceal sign is present (even subtle), prefer label="Varices" with low confidence and grade=1.
- If no variceal sign, choose "Normal".
- If the image does not show the esophagus or is heavily blurred, set quality accordingly but keep label based on what is visible.
- Evidence list entries must be <=3 words each.
- Output N JSON lines only; no extra text.
Diagnostic hints (do not print): serpiginous or beaded/tortuous submucosal veins, blue submucosal hue, red wale marks, cherry red spots, fibrin plug.
"""

def compress_to_cap(raw: bytes, cap_bytes: int, target_side_init: int = 704, quality_init: int = 62):
    if not PIL_OK:
        return raw, ("image/jpeg" if raw[:3] == b"\xff\xd8\xff" else "image/png")
    im = Image.open(BytesIO(raw)).convert("RGB")
    sides = [target_side_init, 640, 576, 512, 448, 384, 320]
    qualities = [quality_init, 58, 55, 50, 45, 40, 36, 32]
    best = None
    for side in sides:
        w, h = im.size
        scale = min(1.0, side / float(max(w, h)))
        im2 = im.resize((max(1,int(w*scale)), max(1,int(h*scale)))) if scale < 1.0 else im
        for q in qualities:
            buf = BytesIO()
            im2.save(buf, format="JPEG", quality=q, optimize=True, subsampling="4:2:0")
            data = buf.getvalue()
            if (best is None) or (len(data) < len(best)):
                best = data
            if len(data) <= cap_bytes:
                return data, "image/jpeg"
    return best if best is not None else raw, "image/jpeg"

def list_subfolders(root: str):
    return [os.path.join(root, d) for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]

def list_images(folder_path: str, sample_every: int):
    files = []
    if not os.path.isdir(folder_path):
        return files
    for fn in sorted(os.listdir(folder_path)):
        if fn.lower().endswith((".jpg",".jpeg",".png",".webp",".bmp")):
            files.append(os.path.join(folder_path, fn))
    if sample_every > 1:
        files = [p for i, p in enumerate(files) if i % sample_every == 0]
    return files

def b64_data_url(mime: str, data: bytes) -> str:
    import base64
    return f"data:{mime};base64,{base64.b64encode(data).decode('utf-8')}"

def build_mapping(names):
    lines = [f"{i+1}:{n}" for i, n in enumerate(names)]
    return "Map:\n" + "\n".join(lines)

def build_parts(mapping_text, items, prompt):
    parts = [{"type": "input_text", "text": prompt},
             {"type": "input_text", "text": mapping_text}]
    for it in items:
        parts.append({"type": "input_image", "image_url": b64_data_url(it["mime"], it["data"])})
    return parts

def call_model(client, parts, max_output_tokens: int, temperature: float, timeout: int = 90) -> str:
    resp = client.responses.create(
        model=OPENAI_MODEL,
        input=[{"role": "user", "content": parts}],
        max_output_tokens=max_output_tokens,
        temperature=temperature,
        timeout=timeout,
    )
    if hasattr(resp, "output_text") and resp.output_text:
        return resp.output_text.strip()
    if hasattr(resp, "output"):
        for out in resp.output:
            if getattr(out, "type","") == "message":
                for c in getattr(out, "content", []):
                    if getattr(c, "type","") == "output_text":
                        return (c.text or "").strip()
    return ""

def parse_jsonl(text: str):
    rows = []
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        try:
            rows.append(json.loads(s))
        except Exception:
            try:
                rows.append(json.loads(s.rstrip(",")))
            except Exception:
                rows.append({"_raw": s, "_error": "json_parse_failed"})
    return rows

def estimate_cost_across(root, sample_every, cap_bytes, estimate_samples, expected_output_tokens_per_image, price_input_per1k, price_output_per1k):
    folders = list_subfolders(root)
    all_files = []
    if not folders:
        all_files.extend(list_images(root, sample_every=sample_every))
    else:
        for folder in folders:
            all_files.extend(list_images(folder, sample_every=sample_every))
    n = len(all_files)
    if n == 0:
        return {"n_images": 0, "avg_bytes": 0, "input_tokens": 0, "input_cost": 0.0, "output_tokens": 0, "output_cost": 0.0, "total_cost": 0.0}
    m = min(estimate_samples, n)
    idxs = [int(i*(n-1)/max(1,m-1)) for i in range(m)]
    total_bytes = 0
    for i in idxs:
        p = all_files[i]
        try:
            with open(p, "rb") as f: raw = f.read()
            data, _ = compress_to_cap(raw, cap_bytes=cap_bytes)
            total_bytes += len(data)
        except Exception:
            total_bytes += cap_bytes
    avg_bytes = total_bytes / m if m else 0
    input_tokens_total = (avg_bytes / 3.0) * n
    input_cost = (input_tokens_total / 1000.0) * price_input_per1k
    output_tokens_total = expected_output_tokens_per_image * n
    output_cost = (output_tokens_total / 1000.0) * price_output_per1k
    total_cost = input_cost + output_cost
    return {"n_images": n, "avg_bytes": avg_bytes, "input_tokens": int(input_tokens_total), "input_cost": input_cost, "output_tokens": int(output_tokens_total), "output_cost": output_cost, "total_cost": total_cost}

def project_running_cost(processed_images, processed_bytes, total_images, expected_output_tokens_per_image, price_input_per1k, price_output_per1k):
    avg_bytes = (processed_bytes/processed_images) if processed_images else 0.0
    projected_input_tokens_total = (avg_bytes/3.0) * total_images
    projected_input_cost_total = (projected_input_tokens_total/1000.0) * price_input_per1k
    projected_output_cost_total = ((expected_output_tokens_per_image*total_images)/1000.0) * price_output_per1k
    return {"avg_bytes": avg_bytes, "projected_total_cost": projected_input_cost_total + projected_output_cost_total, "projected_input_cost": projected_input_cost_total, "projected_output_cost": projected_output_cost_total}

def pack_by_size_min_chunks(images, max_bytes, max_images):
    chunks, cur, cur_b = [], [], 0
    for im in images:
        if (not cur) or (len(cur) < max_images and (cur_b + im["size"]) <= max_bytes):
            cur.append(im); cur_b += im["size"]
        else:
            chunks.append(cur); cur=[im]; cur_b = im["size"]
    if cur: chunks.append(cur)
    return chunks

def run_batch(client, batch, names, expected_output_tokens_per_image, max_output_cap, temperature):
    mapping = build_mapping(names)
    parts = build_parts(mapping, batch, PROMPT_CLASSIFY)
    max_tokens = min(max_output_cap, max(200, expected_output_tokens_per_image * len(batch)))
    try:
        text = call_model(client, parts, max_output_tokens=max_tokens, temperature=temperature)
        status = "ok" if text else "empty"
        return text, status
    except Exception as e:
        return str(e), "error"

def classify_with_shrink(client, batch, names, expected_output_tokens_per_image, max_output_cap, temperature, call_delay):
    if len(batch) <= MIN_CHUNK_IMAGES:
        texts = []
        for im in batch:
            t, st = run_batch(client, [im], [im["name"]], expected_output_tokens_per_image, max_output_cap, temperature)
            texts.append(t if st=="ok" else "")
            time.sleep(call_delay)
        return "\n".join(texts), "ok"
    text, status = run_batch(client, batch, names, expected_output_tokens_per_image, max_output_cap, temperature)
    if status == "ok" and text:
        return text, status
    mid = max(1, len(batch)//2)
    left, right = batch[:mid], batch[mid:]
    t1, s1 = classify_with_shrink(client, left, names[:len(left)], expected_output_tokens_per_image, max_output_cap, temperature, call_delay)
    t2, s2 = classify_with_shrink(client, right, names[len(left):], expected_output_tokens_per_image, max_output_cap, temperature, call_delay)
    return "\n".join([t1, t2]).strip(), "ok" if (s1=="ok" or s2=="ok") else "error"

def summarize_folder_decision(per_image_rows, var_prop_threshold, min_ok_frames, min_var_frames, min_conf_count):
    ok_rows = [r for r in per_image_rows if r.get("quality") == "OK"]
    conf_ok = [r for r in ok_rows if int(r.get("confidence",0)) >= min_conf_count]
    ok_count = len(conf_ok)
    var_rows = [r for r in conf_ok if r.get("label") == "Varices"]
    var_prop = (len(var_rows) / ok_count) if ok_count else 0.0
    note = []
    if ok_count < min_ok_frames:
        note.append(f"Low-OK-frames ({ok_count} < {min_ok_frames})")
    if (var_prop >= var_prop_threshold) and (ok_count >= min_ok_frames) and (len(var_rows) >= min_var_frames):
        grades = [int(r.get("grade", 0) or 0) for r in var_rows]
        try:
            grade = stats.mode(grades)
        except Exception:
            from statistics import median
            grade = int(round(median(grades))) if grades else 1
        label = f"Esophageal Varices Grade {grade}"
        grade_num = grade
    else:
        label = "Normal"
        grade_num = 0
        if var_prop > 0:
            note.append(f"Varices frames present but below threshold ({var_prop:.1%})")
    qc = "; ".join(note) if note else "OK"
    return label, var_prop, ok_count, qc, grade_num

def process_folder(client, folder_path, cap_bytes, sample_every, expected_output_tokens_per_image, max_output_cap, temperature, live_estimate, run_bytes_acc, run_count_acc, max_images_steps, call_delay):
    folder_name = os.path.basename(folder_path.rstrip("\\/"))
    files = list_images(folder_path, sample_every=sample_every)
    images = []
    for p in files:
        with open(p, "rb") as f: raw = f.read()
        data, mime = compress_to_cap(raw, cap_bytes=cap_bytes)
        images.append({"name": os.path.basename(p), "data": data, "mime": mime, "size": len(data)})
    if not images:
        return folder_name, "Tidak ada gambar", [], [], []
    batches = []
    total_bytes = sum(im["size"] for im in images)
    if SINGLE_REQUEST_IF_POSSIBLE and len(images) <= max_images_steps[0] and total_bytes <= MAX_REQUEST_BYTES:
        batches = [images]
    else:
        batches = pack_by_size_min_chunks(images, MAX_REQUEST_BYTES, max_images_steps[0])
    current_step_idx = 0
    current_max_images = max_images_steps[current_step_idx]
    consecutive_fail = 0
    detail_rows, per_image_rows, batch_reports = [], [], []
    i = 0
    while i < len(batches):
        batch = batches[i]
        if len(batch) > current_max_images:
            sub = pack_by_size_min_chunks(batch, MAX_REQUEST_BYTES, current_max_images)
            batches.pop(i)
            for k, sb in enumerate(sub):
                batches.insert(i + k, sb)
            print(f"Repack: pecah batch menjadi {len(sub)} bagian (max {current_max_images} img/chunk)."
                 )
            continue
        mb = sum(im['size'] for im in batch) / (1024 * 1024)
        print(f"Chunk {i+1}/{len(batches)}: {len(batch)} gambar, {mb:.2f} MB")
        time.sleep(call_delay)
        names = [im["name"] for im in batch]
        text, status = classify_with_shrink(client, batch, names, expected_output_tokens_per_image, max_output_cap, temperature, call_delay)
        if live_estimate:
            batch_bytes = sum(im["size"] for im in batch)
            run_bytes_acc["val"] += batch_bytes
            run_count_acc["val"] += len(batch)
            proj = project_running_cost(run_count_acc["val"], run_bytes_acc["val"], run_count_acc["total"], expected_output_tokens_per_image, PRICE_INPUT_PER1K, PRICE_OUTPUT_PER1K)
            print(f"[LIVE] {folder_name} Proc={run_count_acc['val']}/{run_count_acc['total']} | avg={proj['avg_bytes']/1024:.1f} KB | Est total=${proj['projected_total_cost']:.2f} (input ${proj['projected_input_cost']:.2f} + output ${proj['projected_output_cost']:.2f})")
        rows = parse_jsonl(text) if isinstance(text, str) else []
        while len(rows) < len(batch): rows.append({})
        for j, im in enumerate(batch):
            obj = rows[j] if j < len(rows) else {}
            try: g = int(obj.get("grade", 0))
            except Exception: g = 0
            try: c = int(obj.get("confidence", 0))
            except Exception: c = 0
            per_image_rows.append({"Folder": folder_name, "Image": im["name"], "label": obj.get("label","Unknown"), "grade": g, "confidence": c, "quality": obj.get("quality","Unknown"), "evidence": "; ".join(obj.get("evidence", [])) if isinstance(obj.get("evidence", []), list) else str(obj.get("evidence",""))})
        detail_rows.append({"Folder": folder_name, "Batch": i+1, "Jumlah Gambar": len(batch), "Ukuran Batch (MB)": f"{mb:.2f}", "Status": "ok" if status=="ok" else status})
        batch_reports.append({"Folder": folder_name, "Batch": i+1, "Jumlah Gambar": len(batch), "Ukuran Batch (MB)": f"{mb:.2f}", "Raw Output": (text[:1000] if isinstance(text,str) else str(text))})
        if status != "ok":
            consecutive_fail += 1
            if consecutive_fail >= ERROR_LOWERING_THRESHOLD and current_step_idx < len(max_images_steps)-1:
                current_step_idx += 1
                current_max_images = max_images_steps[current_step_idx]
                print(f"Lower step: consecutive fails >= {ERROR_LOWERING_THRESHOLD}. max_images → {current_max_images}")
                consecutive_fail = 0
        else:
            consecutive_fail = 0
        i += 1
    label, var_prop, ok_count, qc_note, grade_num = summarize_folder_decision(per_image_rows, VARICES_PROP_THRESHOLD, MIN_OK_QUALITY_FRAMES, MIN_VAR_FRAMES_REQUIRED, MIN_CONF_COUNT)
    per_image_rows.append({"Folder": folder_name, "Image": "__FOLDER_SUMMARY__", "label": label, "grade": (grade_num if grade_num else None), "confidence": None, "quality": f"OK-frames={ok_count}; var_prop={var_prop:.1%}", "evidence": qc_note})
    return folder_name, label, detail_rows, per_image_rows, batch_reports

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out-excel", required=True)
    ap.add_argument("--api_key", default=os.environ.get("OPENAI_API_KEY",""))
    ap.add_argument("--target-kb", type=int, default=TARGET_KB)
    ap.add_argument("--sample-every", type=int, default=SAMPLE_EVERY)
    ap.add_argument("--chunk-steps", type=str, default="30,20,10")
    ap.add_argument("--min-chunk-images", type=int, default=MIN_CHUNK_IMAGES)
    ap.add_argument("--call-delay", type=float, default=CALL_DELAY_SECONDS)
    ap.add_argument("--error-lowering-threshold", type=int, default=ERROR_LOWERING_THRESHOLD)
    ap.add_argument("--single-request", action="store_true")
    ap.add_argument("--expected-output-tokens", type=int, default=EXPECTED_OUTPUT_TOKENS)
    ap.add_argument("--max-output-cap", type=int, default=MAX_OUTPUT_CAP)
    ap.add_argument("--temperature", type=float, default=TEMPERATURE)
    ap.add_argument("--budget-usd", type=float, default=BUDGET_USD)
    ap.add_argument("--estimate-samples", type=int, default=ESTIMATE_SAMPLES)
    ap.add_argument("--price-input", type=float, default=PRICE_INPUT_PER1K)
    ap.add_argument("--price-output", type=float, default=PRICE_OUTPUT_PER1K)
    ap.add_argument("--allow-overbudget", action="store_true")
    ap.add_argument("--live-estimate", action="store_true")
    ap.add_argument("--var-prop-threshold", type=float, default=VARICES_PROP_THRESHOLD)
    ap.add_argument("--min-ok-frames", type=int, default=MIN_OK_QUALITY_FRAMES)
    ap.add_argument("--min-var-frames", type=int, default=MIN_VAR_FRAMES_REQUIRED)
    ap.add_argument("--min-conf-count", type=int, default=MIN_CONF_COUNT)
    args = ap.parse_args()
    max_images_steps = [int(x) for x in args.chunk_steps.split(",") if x.strip()]
    call_delay = args.call_delay
    if not OpenAI:
        print("Error: openai SDK v1 tidak terpasang. `pip install openai>=1.40`"); sys.exit(2)
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY") or ""
    if not api_key:
        api_key = input("Masukkan OPENAI_API_KEY: ").strip()
    if not api_key:
        print("API key wajib diisi."); sys.exit(2)
    client = OpenAI(api_key=api_key)
    out_dir = os.path.dirname(args.out_excel) or "."
    os.makedirs(out_dir, exist_ok=True)
    cap_bytes = args.target_kb * 1024
    est = estimate_cost_across(args.input, args.sample_every, cap_bytes, args.estimate_samples, args.expected_output_tokens, args.price_input, args.price_output)
    print("=== ESTIMASI BIAYA ===")
    print(f"Gambar diproses : {est['n_images']}")
    print(f"Rata-rata bytes : {int(est['avg_bytes'])} B (~{est['avg_bytes']/1024:.1f} KB) sebelum base64")
    print(f"Input tokens     : {est['input_tokens']:,}")
    print(f"Biaya input      : ${est['input_cost']:.2f}")
    print(f"Output tokens    : {est['output_tokens']:,}")
    print(f"Biaya output     : ${est['output_cost']:.2f}")
    print(f"TOTAL ESTIMASI   : ${est['total_cost']:.2f}  (Budget: ${args.budget_usd:.2f})\n")
    if (not args.allow_overbudget) and (est["total_cost"] > args.budget_usd):
        print("❌ Estimasi melebihi budget. Sesuaikan parameter (--target-kb, --sample-every) atau pakai --allow-overbudget")
        sys.exit(0)
    folders = list_subfolders(args.input)
    if not folders:
        if list_images(args.input, sample_every=args.sample_every):
            folders = [args.input]
        else:
            print("Tidak ada subfolder/gambar cocok di input."); sys.exit(0)
    total_imgs = sum(len(list_images(f, args.sample_every)) for f in folders)
    run_bytes_acc = {"val": 0}
    run_count_acc = {"val": 0, "total": total_imgs}
    print(f"Mulai proses {len(folders)} folder | total gambar (sampling every {args.sample_every}) = {total_imgs}\n")
    summary_rows, all_details, all_per_image, all_batch_reports = [], [], [], []
    for folder in folders:
        try:
            folder_name, final_label, details, per_rows, reports = process_folder(client, folder, cap_bytes, args.sample_every, args.expected_output_tokens, args.max_output_cap, args.temperature, args.live_estimate, run_bytes_acc, run_count_acc, max_images_steps=max_images_steps, call_delay=call_delay)
            label, var_prop, ok_count, qc_note, grade_num = summarize_folder_decision([r for r in per_rows if r["Image"] != "__FOLDER_SUMMARY__"], args.var_prop_threshold, args.min_ok_frames, args.min_var_frames, args.min_conf_count)
            summary_rows.append({"Folder": folder_name, "Hasil OpenAI": label, "Grade": (grade_num if grade_num else ""), "OK_Frames": ok_count, "Var_Prop%": f"{var_prop*100:.1f}%", "Status": "Success"})
            all_details.extend(details); all_per_image.extend(per_rows); all_batch_reports.extend(reports)
        except Exception as e:
            summary_rows.append({"Folder": os.path.basename(folder.rstrip('\\/')), "Hasil OpenAI": f"Error: {e}", "Grade":"", "OK_Frames":"", "Var_Prop%":"", "Status": "Error"})
            print(f"Error folder {folder}: {e}"); continue
    out_path = args.out_excel
    try:
        with pd.ExcelWriter(out_path, engine="xlsxwriter") as writer:
            pd.DataFrame(summary_rows).to_excel(writer, sheet_name="Ringkasan", index=False)
            pd.DataFrame(all_details).to_excel(writer, sheet_name="Detail Batch", index=False)
            pd.DataFrame(all_per_image).to_excel(writer, sheet_name="Per-Image", index=False)
            pd.DataFrame(all_batch_reports).to_excel(writer, sheet_name="Batch Raw", index=False)
    except PermissionError:
        base, ext = os.path.splitext(out_path); alt = f"{base}_1{ext}"
        with pd.ExcelWriter(alt, engine="xlsxwriter") as writer:
            pd.DataFrame(summary_rows).to_excel(writer, sheet_name="Ringkasan", index=False)
            pd.DataFrame(all_details).to_excel(writer, sheet_name="Detail Batch", index=False)
            pd.DataFrame(all_per_image).to_excel(writer, sheet_name="Per-Image", index=False)
            pd.DataFrame(all_batch_reports).to_excel(writer, sheet_name="Batch Raw", index=False)
        out_path = alt
    print(f"Selesai. Excel: {out_path}")

if __name__ == "__main__":
    main()
