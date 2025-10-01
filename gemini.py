import os
import time
import json
import math
import statistics as stats
import pandas as pd
from tqdm import tqdm
import google.generativeai as genai

# ====== KONFIGURASI YANG BISA ANDA UBAH ======
# Pakai environment variable kalau ada, kalau kosong nanti diminta saat runtime
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

FOLDER_INPUT = r"D:\Pediatri\PIT\endos"
OUTPUT_EXCEL = r"D:\Pediatri\PIT\hasil_gemini.xlsx"

# Batas aman payload per request (single request per folder jika total ukuran <= 17 MB)
MAX_REQUEST_BYTES = 17 * 1024 * 1024  # 17 MB

# Mode kirim tunggal per folder jika memungkinkan
SINGLE_REQUEST_IF_POSSIBLE = True

# Skip behavior
SKIP_IF_ALREADY_PROCESSED = True   # skip folder yang sudah "Success" di Excel
SKIP_IF_ALREADY_LISTED = False     # kalau True, skip semua yang sudah tercatat (termasuk Error)

# Kontrol chunk & fallback (awal)
MAX_IMAGES_PER_CHUNK_STEPS = [30, 20, 10]  # lebih rendah agar output per-image stabil
SHRINK_ON_FAILURE = True           # gagal → pecah dua (binary split), bukan per‑gambar langsung
MIN_CHUNK_IMAGES = 5               # batas bawah sebelum akhirnya per‑gambar
CALL_DELAY_SECONDS = 0.7           # jeda antar panggilan API (hemat kuota & stabil)
ERROR_LOWERING_THRESHOLD = 3       # jika error/per-gambar beruntun >= ini → turunkan step

# Ambang keputusan folder (hemat false negative)
VARICES_PROP_THRESHOLD = 0.03  # minimal 3% frame berkualitas OK bertanda varises
MIN_OK_QUALITY_FRAMES = 5     # minimal frame OK agar keputusan folder dianggap valid

# Prompt klasifikasi per-image (output JSON baris per gambar)
PROMPT_CLASSIFY = (
    "You are a pediatric gastroenterology expert. You will receive N endoscopy images in order.\n"
    "FIRST, read this mapping from index to filename.\n"
    "THEN, for EACH image 1..N, output EXACTLY one compact JSON on its own line, no extra text, no code fences.\n"
    '{"image":"<filename>","label":"Normal|Varices","grade":0-3,'
    '"confidence":0-100,"quality":"OK|Blurred|OutOfEsophagus","evidence":["..."]}'
    "\nRules: If any variceal sign is present but subtle, prefer label=\"Varices\" with low confidence and grade=1.\n"
    "If the image does not show esophagus or is heavily blurred, set quality accordingly and keep label based on what is visible.\n"
    "Diagnostic hints (do not print): serpiginous/blue submucosal columns in distal esophagus (palisade zone), beaded/tortuous veins, red wale marks, cherry red spots, fibrin plug.\n"
    "Output N JSON lines only."
)

# Stabilitas output
GENERATION_CONFIG = {
    "temperature": 0.05,
    "max_output_tokens": 2048,
}

# Model utama & cadangan (diset sesuai preferensi Anda)
PRIMARY_MODEL = "gemini-2.0-flash"
FALLBACK_MODEL = "gemini-2.5-pro"
USE_FALLBACK_MODEL = False  # hemat kuota: nonaktifkan fallback kecuali diperlukan
# ============================================


# ====== SETUP MODEL ======
_model_cache = {}

def ensure_api_key():
    """Pastikan API key terisi; jika kosong, minta input dari user."""
    global GEMINI_API_KEY
    while not GEMINI_API_KEY:
        key = input("Masukkan GEMINI_API_KEY: ").strip()
        if key:
            GEMINI_API_KEY = key
        else:
            print("API key wajib diisi untuk melanjutkan.")
    genai.configure(api_key=GEMINI_API_KEY)


def get_model(model_name: str):
    """Cache model agar tidak re-instantiate berulang."""
    if model_name not in _model_cache:
        _model_cache[model_name] = genai.GenerativeModel(model_name, generation_config=GENERATION_CONFIG)
    return _model_cache[model_name]
# =========================


def _response_to_text(resp):
    """Ambil teks aman dari respons Gemini, atau kembalikan None jika tidak ada."""
    try:
        if hasattr(resp, "text") and resp.text:
            return resp.text.strip()
    except Exception:
        pass
    try:
        if getattr(resp, "candidates", None):
            for c in resp.candidates:
                content = getattr(c, "content", None)
                parts = getattr(content, "parts", None) if content else None
                if parts:
                    texts = [getattr(p, "text", None) for p in parts]
                    texts = [t for t in texts if t]
                    if texts:
                        return "\n".join(texts).strip()
    except Exception:
        pass
    return None


def robust_json_extract(text):
    """Ekstrak list dict JSON dari teks: dukung satu JSON per baris ATAU banyak objek dalam satu paragraf."""
    items = []
    if not text:
        return items
    # coba per baris
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                items.append(obj)
                continue
        except Exception:
            pass
    # jika masih sedikit, coba regex cari {...}
    if len(items) <= 1:
        import re
        for m in re.finditer(r"\{[^{}]+\}", text, flags=re.DOTALL):
            try:
                obj = json.loads(m.group(0))
                if isinstance(obj, dict):
                    items.append(obj)
            except Exception:
                continue
    return items


def classify_with_model(model_name, batch, index2fname, retries=2, backoff=2.0):
    """Kirim satu batch dengan model tertentu. Kembalikan teks atau string error khusus."""
    global GEMINI_API_KEY

    # Siapkan mapping teks agar model tahu urutan
    mapping_lines = [f"Image {i+1} = {fn}" for i, fn in enumerate(index2fname)]
    mapping_text = "\n".join(["Mapping:"] + mapping_lines + [f"N = {len(index2fname)}"])

    parts = [
        {"text": PROMPT_CLASSIFY},
        {"text": mapping_text},
    ]
    for img in batch:
        parts.append({"inline_data": {"mime_type": img["mime_type"], "data": img["data"]}})

    model = get_model(model_name)

    for attempt in range(1, retries + 1):
        try:
            resp = model.generate_content([{"role": "user", "parts": parts}])
            text = _response_to_text(resp)
            if text:
                return text
            break  # respons kosong -> hentikan loop untuk fallback di level di atas
        except Exception as e:
            msg = str(e).lower()

            # Quota/limit -> minta API key baru
            if any(k in msg for k in ["quota", "limit", "exceeded"]):
                print("\n⚠️ Quota API key habis atau limit tercapai.")
                new_key = input("Masukkan API key baru untuk melanjutkan (kosong untuk stop): ").strip()
                if new_key:
                    GEMINI_API_KEY = new_key
                    genai.configure(api_key=GEMINI_API_KEY)
                    _model_cache.clear()  # reset cache karena config berubah
                    print("🔑 API key berhasil diperbarui, coba ulang batch ini...")
                    continue
                else:
                    print("❌ API key baru tidak diberikan, proses dihentikan.")
                    raise e

            # Error sementara -> retry dengan backoff
            if any(k in msg for k in ["timeout", "temporarily", "unavailable", "internal", "503", "502", "500"]):
                if attempt < retries:
                    sleep_s = backoff ** (attempt - 1)
                    print(f"⏳ Error sementara ({model_name}): {e}. Retry {attempt}/{retries} setelah {sleep_s:.1f}s...")
                    time.sleep(sleep_s)
                    continue

            # Error lain -> hentikan loop untuk fallback di level di atas
            break

    return "Error: empty response or invalid parts"


def classify_batch_with_fallback(batch, index2fname, retries=2, backoff=2.0):
    """
    Coba model utama dulu. Kalau respons kosong/invalid atau error 5xx/timeout,
    dan USE_FALLBACK_MODEL=True, coba ulang dengan model cadangan.
    """
    txt = classify_with_model(PRIMARY_MODEL, batch, index2fname, retries=retries, backoff=backoff)
    if txt and not txt.lower().startswith("error: empty response"):
        return txt

    if USE_FALLBACK_MODEL:
        print(f"❗ Model utama gagal/kosong. Ganti ke {FALLBACK_MODEL} untuk chunk ini...")
        txt2 = classify_with_model(FALLBACK_MODEL, batch, index2fname, retries=retries, backoff=backoff)
        if txt2 and not txt2.lower().startswith("error: empty response"):
            return txt2

    return "Error: empty response or invalid parts"


def classify_with_shrink(batch, index2fname, retries=2, backoff=2.0):
    """
    Coba kirim chunk apa adanya (primary → fallback). Jika gagal (kosong/invalid),
    lakukan binary split sampai ukuran kecil; baru per‑gambar sebagai langkah terakhir.
    Mengembalikan (text, status):
      - status: "ok" | "shrink" | "per_image" | "error"
    """
    txt = classify_batch_with_fallback(batch, index2fname, retries=retries, backoff=backoff)
    if txt and not txt.lower().startswith("error: empty response"):
        return txt, "ok"

    if not SHRINK_ON_FAILURE:
        # langsung per‑gambar (paling boros; dihindari)
        print("ℹ️ Fallback per‑gambar...")
        parts = []
        for i, im in enumerate(batch, 1):
            time.sleep(CALL_DELAY_SECONDS)
            t = classify_batch_with_fallback([im], [index2fname[i-1]], retries=retries, backoff=backoff)
            parts.append(t if t else "")
        return "\n".join(parts), "per_image"

    # shrink: pecah jadi dua bagian
    if len(batch) <= MIN_CHUNK_IMAGES:
        print("ℹ️ Gagal di chunk kecil. Fallback per‑gambar (terakhir).")
        parts = []
        for i, im in enumerate(batch, 1):
            time.sleep(CALL_DELAY_SECONDS)
            t = classify_batch_with_fallback([im], [index2fname[i-1]], retries=retries, backoff=backoff)
            parts.append(t if t else "")
        return "\n".join(parts), "per_image"

    mid = max(1, len(batch) // 2)
    left, right = batch[:mid], batch[mid:]
    left_map, right_map = index2fname[:mid], index2fname[mid:]
    print(f"↘️ Shrink chunk: {len(batch)} → {len(left)} + {len(right)}")
    time.sleep(CALL_DELAY_SECONDS)
    left_txt, left_status  = classify_with_shrink(left, left_map, retries=retries, backoff=backoff)
    time.sleep(CALL_DELAY_SECONDS)
    right_txt, right_status = classify_with_shrink(right, right_map, retries=retries, backoff=backoff)
    status = "shrink" if ("error" not in (left_status, right_status) and "ok" not in (left_status, right_status)) else "ok"
    combined = (left_txt or "") + ("\n" if left_txt and right_txt else "") + (right_txt or "")
    return combined, status


def read_images_from_folder(folder_path):
    images = []
    for file in sorted(os.listdir(folder_path)):
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            ext = os.path.splitext(file)[1].lower()
            mime_type = "image/png" if ext == ".png" else "image/jpeg"
            with open(os.path.join(folder_path, file), "rb") as f:
                data = f.read()
            images.append({"mime_type": mime_type, "data": data, "name": file, "size": len(data)})
    return images


def quick_quality_filter(images, enable=True):
    """Filter cepat (opsional) untuk menandai gambar terlalu gelap/terang/kontras rendah. Tidak membuang, hanya flag lokal.
    Memerlukan Pillow. Jika tidak ada Pillow, akan dilewati tanpa error."""
    flags = {img["name"]: {"too_dark": False, "too_bright": False, "low_contrast": False} for img in images}
    if not enable:
        return flags
    try:
        from PIL import Image
        from io import BytesIO
        for img in images:
            try:
                im = Image.open(BytesIO(img["data"]))
                im = im.convert("L").resize((128, 128))
                px = list(im.getdata())
                mean = sum(px) / len(px)
                var = sum((p - mean) ** 2 for p in px) / len(px)
                std = math.sqrt(var)
                flags[img["name"]]["too_dark"] = mean < 25
                flags[img["name"]]["too_bright"] = mean > 230
                flags[img["name"]]["low_contrast"] = std < 10
            except Exception:
                continue
    except Exception:
        # Pillow tidak tersedia, abaikan filter
        return flags
    return flags


def pack_by_size_min_chunks(images, max_bytes, max_images):
    """
    Bagi list images menjadi chunk sesedikit mungkin:
    - total size per chunk <= max_bytes
    - jumlah gambar per chunk <= max_images
    (Greedy, urut sesuai input.)
    """
    batches = []
    current, size_now = [], 0

    for img in images:
        img_size = img["size"]
        if (not current) or (size_now + img_size <= max_bytes and len(current) < max_images):
            current.append(img); size_now += img_size
        else:
            batches.append(current)
            current = [img]; size_now = img_size

    if current:
        batches.append(current)
    return batches


def summarize_folder_decision(per_image_rows):
    """Tentukan label folder berbasis proporsi dan kualitas.
    - Gunakan hanya frame quality == OK saat menghitung proporsi varices.
    - Varices jika proporsi >= VARICES_PROP_THRESHOLD dan MIN_OK_QUALITY_FRAMES tercapai. Grade = modus (fallback median).
    - Jika tidak memenuhi, Default Normal tetapi tetap cantumkan catatan QC.
    """
    ok_rows = [r for r in per_image_rows if r.get("quality") == "OK"]
    ok_count = len(ok_rows)
    var_rows = [r for r in ok_rows if r.get("label") == "Varices"]
    var_prop = (len(var_rows) / ok_count) if ok_count else 0.0

    note = []
    if ok_count < MIN_OK_QUALITY_FRAMES:
        note.append(f"Low-OK-frames ({ok_count} < {MIN_OK_QUALITY_FRAMES})")

    if var_prop >= VARICES_PROP_THRESHOLD and ok_count >= MIN_OK_QUALITY_FRAMES and len(var_rows) > 0:
        grades = [int(r.get("grade", 0) or 0) for r in var_rows]
        try:
            grade = stats.mode(grades)
        except Exception:
            grade = int(round(stats.median(grades))) if grades else 1
        label = f"Esophageal Varices Grade {grade}"
    else:
        label = "Normal"
        if var_prop > 0:
            note.append(f"Varices frames present but below threshold ({var_prop:.1%})")

    qc = "; ".join(note) if note else "OK"
    return label, var_prop, ok_count, qc


def process_folder(folder_path):
    folder_name = os.path.basename(folder_path)
    images = read_images_from_folder(folder_path)

    if not images:
        return folder_name, "Tidak ada gambar", [], []

    # QC cepat (lokal)
    qc_flags = quick_quality_filter(images, enable=True)

    total_bytes = sum(img["size"] for img in images)

    # urutan batas gambar per chunk yang akan digunakan (dinamis menurun jika error)
    current_step_idx = 0
    current_max_images = MAX_IMAGES_PER_CHUNK_STEPS[current_step_idx]

    # single request hanya jika muat size & tidak melebihi batas gambar per chunk
    if SINGLE_REQUEST_IF_POSSIBLE and total_bytes <= MAX_REQUEST_BYTES and len(images) <= current_max_images:
        batches = [images]
        mode = "single-request"
    else:
        batches = pack_by_size_min_chunks(images, MAX_REQUEST_BYTES, current_max_images)
        mode = f"multi-chunk ({len(batches)} parts)"

    detail_rows, per_image_rows, batch_reports = [], [], []
    print(f"[{folder_name}] Mode: {mode}. Total gambar: {len(images)}. Total ukuran: {total_bytes/1024/1024:.2f} MB")
    if len(batches) > 1:
        print(f"[{folder_name}] {len(batches)} chunk akan diproses.")

    # untuk menurunkan chunk size jika error beruntun
    consecutive_fail = 0
    i = 0
    img_global_idx = 0
    while i < len(batches):
        batch = batches[i]
        # jika batch ini melebihi current_max_images (karena batas diturunkan), pecah dulu
        if len(batch) > current_max_images:
            sub = pack_by_size_min_chunks(batch, MAX_REQUEST_BYTES, current_max_images)
            # ganti batch i dengan sub-batches
            batches.pop(i)
            for k, sb in enumerate(sub):
                batches.insert(i + k, sb)
            print(f"🔧 Repack batch karena limit turun: dipecah jadi {len(sub)} bagian (max {current_max_images} img/chunk).")
            continue  # proses batch baru pada index i

        mb = sum(img['size'] for img in batch) / (1024 * 1024)
        print(f"  - Chunk {i+1}/{len(batches)}: {len(batch)} gambar, {mb:.2f} MB")
        time.sleep(CALL_DELAY_SECONDS)  # throttle antar panggilan

        # siapkan mapping index->filename untuk batch ini
        index2fname = [img["name"] for img in batch]
        text, status = classify_with_shrink(batch, index2fname)
        batch_reports.append({
            "Folder": folder_name,
            "Batch": i + 1,
            "Jumlah Gambar": len(batch),
            "Ukuran Batch (MB)": f"{mb:.2f}",
            "Raw Output": text[:1000] if isinstance(text, str) else str(text)  # simpan cuplikan agar Excel ringan
        })

        # parse hasil JSON per-image
        objs = robust_json_extract(text)
        # Normalisasi panjang: kalau kurang dari jumlah gambar, kita isi placeholder agar tidak shift
        while len(objs) < len(batch):
            objs.append({})
        for j, img in enumerate(batch):
            obj = objs[j] if j < len(objs) else {}
            name = img["name"]
            qc = qc_flags.get(name, {})
            row = {
                "Folder": folder_name,
                "Image": name,
                "IndexGlobal": img_global_idx + 1,
                "label": obj.get("label", "Unknown"),
                "grade": int(obj.get("grade", 0) or 0) if isinstance(obj.get("grade", 0), (int, float, str)) else 0,
                "confidence": int(obj.get("confidence", 0) or 0) if isinstance(obj.get("confidence", 0), (int, float, str)) else 0,
                "quality": obj.get("quality", "Unknown"),
                "evidence": "; ".join(obj.get("evidence", [])) if isinstance(obj.get("evidence", []), list) else str(obj.get("evidence", "")),
                "pre_qc_too_dark": qc.get("too_dark", False),
                "pre_qc_too_bright": qc.get("too_bright", False),
                "pre_qc_low_contrast": qc.get("low_contrast", False),
            }
            per_image_rows.append(row)
            img_global_idx += 1

        detail_rows.append({
            "Folder": folder_name,
            "Batch": i + 1,
            "Jumlah Gambar": len(batch),
            "Ukuran Batch (MB)": f"{mb:.2f}",
            "Status": status
        })

        # kelola penurunan batas jika sering gagal
        if status in ("error", "per_image"):
            consecutive_fail += 1
        else:
            consecutive_fail = 0

        if consecutive_fail >= ERROR_LOWERING_THRESHOLD and current_step_idx < len(MAX_IMAGES_PER_CHUNK_STEPS) - 1:
            current_step_idx += 1
            new_limit = MAX_IMAGES_PER_CHUNK_STEPS[current_step_idx]
            print(f"⚠️ Terlalu banyak error berturut-turut. Menurunkan MAX_IMAGES_PER_CHUNK menjadi {new_limit}.")
            current_max_images = new_limit
            # Repack sisa batch (setelah i) dengan limit baru
            remaining_imgs = [img for b in batches[i+1:] for img in b]
            new_tail = pack_by_size_min_chunks(remaining_imgs, MAX_REQUEST_BYTES, current_max_images)
            batches = batches[:i+1] + new_tail
            print(f"🔧 Sisa proses di-repack menjadi {len(new_tail)} chunk (max {current_max_images} img/chunk).")
            consecutive_fail = 0  # reset setelah menurunkan

        i += 1

    # Keputusan akhir folder
    final_label, var_prop, ok_count, qc_note = summarize_folder_decision(per_image_rows)
    return folder_name, final_label, detail_rows, per_image_rows + [{
        "Folder": folder_name,
        "Image": "__FOLDER_SUMMARY__",
        "IndexGlobal": None,
        "label": final_label,
        "grade": None,
        "confidence": None,
        "quality": f"OK-frames={ok_count}; var_prop={var_prop:.1%}",
        "evidence": qc_note,
        "pre_qc_too_dark": None,
        "pre_qc_too_bright": None,
        "pre_qc_low_contrast": None,
    }], batch_reports


def main():
    ensure_api_key()

    # Muat ringkasan lama (untuk skip)
    processed_success = set()
    processed_all = set()
    if os.path.exists(OUTPUT_EXCEL):
        try:
            df_summary = pd.read_excel(OUTPUT_EXCEL, sheet_name="Ringkasan")
            if "Folder" in df_summary.columns:
                processed_all = set(df_summary["Folder"].astype(str).tolist())
            if {"Folder", "Status"}.issubset(df_summary.columns):
                processed_success = set(df_summary.loc[df_summary["Status"] == "Success", "Folder"].astype(str).tolist())
        except Exception:
            pass

    folders = [os.path.join(FOLDER_INPUT, d) for d in os.listdir(FOLDER_INPUT)
               if os.path.isdir(os.path.join(FOLDER_INPUT, d))]

    total_folders = len(folders)
    processed_count = 0
    success_count = 0
    error_count = 0

    summary_rows, all_details, all_per_image, all_batch_reports = [], [], [], []

    print(f"🔍 Memproses {total_folders} folder (Gemini, batch-per-image JSON, ≤ 17 MB, shrink-on-failure)...\n")
    for folder_path in tqdm(folders, desc="Progress Folder"):
        folder_name = os.path.basename(folder_path)

        # Aturan skip
        if SKIP_IF_ALREADY_LISTED and folder_name in processed_all:
            print(f"⏭️ Skip '{folder_name}' (sudah pernah dianalisis - listed di Excel).")
            continue
        if SKIP_IF_ALREADY_PROCESSED and folder_name in processed_success:
            print(f"⏭️ Skip '{folder_name}' (status Success sudah ada di Excel).")
            continue

        try:
            folder_name, final_res, details, per_imgs, batch_reports = process_folder(folder_path)
            status = "Success" if not str(final_res).lower().startswith("error") else "Error"
            success_count += 1 if status == "Success" else 0
            error_count += 1 if status == "Error" else 0
            processed_count += 1

            summary_rows.append({"Folder": folder_name, "Hasil Gemini": final_res, "Status": status})
            all_details.extend(details)
            all_per_image.extend(per_imgs)
            all_batch_reports.extend(batch_reports)
        except Exception as e:
            error_count += 1
            processed_count += 1
            summary_rows.append({"Folder": folder_name, "Hasil Gemini": f"Error: {e}", "Status": "Error"})
            print(f"❌ Error pada folder '{folder_name}': {e}")
            continue

    # Tulis ke Excel (3 sheet: Ringkasan, Detail Batch, Per-Image + 1 sheet optional Raw Batch)
    if os.path.exists(OUTPUT_EXCEL):
        with pd.ExcelWriter(OUTPUT_EXCEL, mode='a', if_sheet_exists='replace') as writer:
            try:
                old_summary = pd.read_excel(OUTPUT_EXCEL, sheet_name="Ringkasan")
            except Exception:
                old_summary = pd.DataFrame(columns=["Folder", "Hasil Gemini", "Status"])
            new_summary = pd.DataFrame(summary_rows)
            combined_summary = pd.concat([old_summary, new_summary], ignore_index=True).drop_duplicates(subset=["Folder"], keep='last')
            combined_summary.to_excel(writer, sheet_name="Ringkasan", index=False)

            try:
                old_detail = pd.read_excel(OUTPUT_EXCEL, sheet_name="Detail Batch")
            except Exception:
                old_detail = pd.DataFrame(columns=["Folder", "Batch", "Jumlah Gambar", "Ukuran Batch (MB)", "Status"])
            new_detail = pd.DataFrame(all_details)
            combined_detail = pd.concat([old_detail, new_detail], ignore_index=True)
            combined_detail.to_excel(writer, sheet_name="Detail Batch", index=False)

            pd.DataFrame(all_per_image).to_excel(writer, sheet_name="Per-Image", index=False)
            pd.DataFrame(all_batch_reports).to_excel(writer, sheet_name="Batch Raw", index=False)
    else:
        with pd.ExcelWriter(OUTPUT_EXCEL) as writer:
            pd.DataFrame(summary_rows).to_excel(writer, sheet_name="Ringkasan", index=False)
            pd.DataFrame(all_details).to_excel(writer, sheet_name="Detail Batch", index=False)
            pd.DataFrame(all_per_image).to_excel(writer, sheet_name="Per-Image", index=False)
            pd.DataFrame(all_batch_reports).to_excel(writer, sheet_name="Batch Raw", index=False)

    # Ringkasan akhir
    print("\n================ RINGKASAN =================")
    print(f"Total folder        : {total_folders}")
    print(f"Diproses (baru)     : {processed_count}")
    print(f"Berhasil (Success)  : {success_count}")
    print(f"Gagal (Error)       : {error_count}")
    print("===========================================\n")

    input("Tekan Enter untuk keluar...")


if __name__ == "__main__":
    main()
