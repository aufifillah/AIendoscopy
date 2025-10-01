import cv2
import os
import re
from datetime import datetime
from rich.progress import Progress, BarColumn, TimeRemainingColumn, TimeElapsedColumn, TextColumn
import concurrent.futures

# ===== KONFIGURASI =====
video_folder = r"D:\Pediatri\PIT\videos"  # Folder sumber video
output_folder = r"D:\Pediatri\PIT\endos"  # Folder tujuan simpan frame
fps_extract = 1  # Ambil frame setiap 1 detik
log_file = os.path.join(output_folder, "extract_log.txt")
# =======================

os.makedirs(output_folder, exist_ok=True)

def write_log(message):
    """Tulis log ke file & print ke layar"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {message}"
    print(line)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(line + "\n")

def verify_frames(frame_files):
    """Periksa apakah semua frame valid & resolusinya konsisten"""
    if not frame_files:
        return False
    valid_frames = 0
    ref_shape = None
    for f in frame_files:
        img = cv2.imread(f)
        if img is None:
            continue
        if ref_shape is None:
            ref_shape = img.shape
        if img.shape != ref_shape:
            continue
        valid_frames += 1
    return valid_frames == len(frame_files)

def extract_frames(video_file, progress, total_task_id):
    """Ekstrak frame dari satu video (bisa resume)"""
    if not video_file.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        write_log(f"[SKIP] {video_file} (bukan file video)")
        return

    video_path = os.path.join(video_folder, video_file)
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    # Tentukan nama folder output berdasarkan angka pertama di nama file
    numbers = re.findall(r'\d+', video_name)
    if numbers:
        base_folder_name = numbers[0][:8] if len(numbers[0]) >= 8 else numbers[0]
    else:
        base_folder_name = video_name

    video_output_folder = os.path.join(output_folder, base_folder_name)
    os.makedirs(video_output_folder, exist_ok=True)

    # Buka video
    cap = cv2.VideoCapture(video_path)
    video_fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if video_fps == 0:
        write_log(f"[ERROR] {video_file} (Tidak bisa membaca FPS)")
        return

    # Hitung target frame
    frame_interval = max(1, int(video_fps / fps_extract))
    expected_frames = total_frames // frame_interval

    # Cek frame yang sudah ada
    existing_frame_paths = sorted([
        os.path.join(video_output_folder, f)
        for f in os.listdir(video_output_folder)
        if f.startswith(video_name) and f.endswith(".jpg")
    ])

    # Hitung posisi terakhir
    saved = len(existing_frame_paths)

    # Skip jika sudah lengkap & valid
    if saved >= expected_frames:
        if verify_frames(existing_frame_paths):
            progress.update(total_task_id, advance=(expected_frames - saved))
            write_log(f"[SKIP] {video_file} (Lengkap & valid {saved}/{expected_frames} frame)")
            cap.release()
            return
        else:
            for f in existing_frame_paths:
                os.remove(f)
            existing_frame_paths = []
            saved = 0
            write_log(f"[FIXED] {video_file} (Frame corrupt dihapus, ekstrak ulang)")

    # Lanjut dari frame terakhir
    cap.set(cv2.CAP_PROP_POS_FRAMES, saved * frame_interval)
    count = saved * frame_interval

    # Progress bar per video
    video_task_id = progress.add_task(f"[cyan]{video_file}", total=expected_frames, completed=saved)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_interval == 0:
            filename = os.path.join(video_output_folder, f"{video_name}_frame_{saved:04d}.jpg")
            cv2.imwrite(filename, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            saved += 1
            progress.update(video_task_id, advance=1)
            progress.update(total_task_id, advance=1)
        count += 1

    cap.release()
    progress.remove_task(video_task_id)
    write_log(f"[DONE] {video_file} ({saved}/{expected_frames} frame)")

# ===== MAIN SCRIPT =====

# Daftar video
video_list = [vf for vf in os.listdir(video_folder) if vf.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))]

# Hitung total frame semua video untuk progress bar total
total_expected = 0
for vf in video_list:
    cap = cv2.VideoCapture(os.path.join(video_folder, vf))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps > 0:
        frame_interval = max(1, int(fps / fps_extract))
        total_expected += total_frames // frame_interval
    cap.release()

# Progress bar dengan ETA
with Progress(
    TextColumn("[bold green]{task.description}"),
    BarColumn(),
    "[progress.percentage]{task.percentage:>3.0f}%",
    "• ETA:",
    TimeRemainingColumn(),
    "• Elapsed:",
    TimeElapsedColumn(),
) as progress:

    total_task_id = progress.add_task("[white]Total Progress", total=total_expected)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(extract_frames, vf, progress, total_task_id) for vf in video_list]
        for future in concurrent.futures.as_completed(futures):
            pass

write_log("Selesai ekstraksi semua video.")
input("\n✅ Proses selesai. Tekan Enter untuk keluar...")
