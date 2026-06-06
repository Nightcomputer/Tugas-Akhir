import os
import cv2
import csv
import time
import shutil
import numpy as np
import pytesseract
import re
from PIL import Image
from scipy.ndimage import interpolation as inter
from jiwer import cer, wer
from itertools import combinations

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

CONFIG_TESSERACT = "--oem 3 --psm 6 -l ind+eng"
DIR_GAMBAR       = "data/gambar"
DIR_GT           = "data/ground_truth"
DIR_OUTPUT       = "output"
DIR_EVAL         = "output/evaluasi"
CSV_OUTPUT       = "output/hasil_evaluasi.csv"

TEKNIK_LIST = [
    "none",
    "grayscale",
    "clahe",
    "binarize",
    "line_removal",
    "adaptive",
]


# ══════════════════════════════════════════════════════════════════════════════
# 1. TEKNIK-TEKNIK PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Konversi RGB ke grayscale. Selalu dijalankan sebagai langkah pertama."""
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return image.copy()

#CLAHE

def clahe(gray: np.ndarray) -> tuple[np.ndarray, dict]:
    std_val = float(np.std(gray))

    if std_val >= 30:
        return gray, {"applied": False, "std_intensitas": round(std_val, 2)}

    enhancer = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(4, 4))
    result = enhancer.apply(gray)
    return result, {"applied": True, "std_intensitas": round(std_val, 2)}


#Binarisasi

def binarize(gray: np.ndarray) -> np.ndarray:
    gray = cv2.medianBlur(gray, 3)

    return cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        111,
        55
    )


#Line Removal

def detect_table_lines(binary: np.ndarray) -> bool:
   
    h, w = binary.shape

    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (max(10, w // 40), 1))
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(10, h // 40)))

    inverted = ~binary

    line_h = cv2.morphologyEx(inverted, cv2.MORPH_OPEN, kernel_h)
    line_v = cv2.morphologyEx(inverted, cv2.MORPH_OPEN, kernel_v)

    total_line_pixels = np.count_nonzero(line_h) + np.count_nonzero(line_v)
    total_pixels = h * w

    ratio = total_line_pixels / total_pixels
    return ratio > 0.003

def remove_lines(binary: np.ndarray) -> tuple[np.ndarray, dict]:
    if not detect_table_lines(binary):
        return binary, {"applied": False, "reason": "tidak terdeteksi garis"}
    h, w = binary.shape
    inverted = ~binary 
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (max(10, w // 40), 1))
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(10, h // 40)))
    line_h = cv2.morphologyEx(inverted, cv2.MORPH_OPEN, kernel_h)
    line_v = cv2.morphologyEx(inverted, cv2.MORPH_OPEN, kernel_v)
    
    masker_garis = cv2.add(line_h, line_v)
    kernel_tebal = np.ones((3, 3), np.uint8)
    masker_garis = cv2.dilate(masker_garis, kernel_tebal, iterations=1)
    hasil_inv = cv2.bitwise_and(inverted, inverted, mask=cv2.bitwise_not(masker_garis))
    result = ~hasil_inv
    return result, {"applied": True}

# ══════════════════════════════════════════════════════════════════════════════
# 2. PIPELINE ADAPTIF LENGKAP
# ══════════════════════════════════════════════════════════════════════════════

def pipeline_adaptive(image: np.ndarray) -> tuple[np.ndarray, dict]:
    log = {}

    img = to_grayscale(image)
    log["grayscale"] = {"applied": True}

    img, log["clahe"]      = clahe(img)

    img = binarize(img)
    log["binarize"] = {"applied": True, "method": "adaptive_gaussian"}

    img, log["line_removal"] = remove_lines(img)

    return img, log


# ══════════════════════════════════════════════════════════════════════════════
# 3. PENGUJIAN PER TEKNIK
# ══════════════════════════════════════════════════════════════════════════════

def process_single_technique(image: np.ndarray, technique: str) -> np.ndarray:
    """
    Jalankan HANYA satu teknik preprocessing.
    Digunakan untuk pengujian individual masing-masing teknik.
    """
    if technique == "none":
        return image.copy()

    elif technique == "grayscale":
        return to_grayscale(image)

    elif technique == "clahe":
        img = to_grayscale(image)
        img, _ = clahe(img)
        return img

    elif technique == "binarize":
        img = to_grayscale(image)
        return binarize(img)

    elif technique == "line_removal":
        img = to_grayscale(image)
        img = binarize(img)
        img, _ = remove_lines(img)
        return img

    elif technique == "adaptive":
        img, _ = pipeline_adaptive(image)
        return img

    else:
        raise ValueError(f"Teknik tidak dikenal: '{technique}'")


# ══════════════════════════════════════════════════════════════════════════════
# 4. OCR
# ══════════════════════════════════════════════════════════════════════════════

def run_ocr(image: np.ndarray) -> dict:
    pil_img = Image.fromarray(image)
    
    text = pytesseract.image_to_string(pil_img, config=CONFIG_TESSERACT)
    text = text.replace("|", "")
    text = re.sub(r'\n\s*\n+', '\n', text)
    text = "\n".join(line.strip() for line in text.splitlines())

    word_count = len([w for w in text.split() if w.strip()])

    return {"text": text.strip(), "word_count": word_count}

# ══════════════════════════════════════════════════════════════════════════════
# 5. PERHITUNGAN CER & WER
# ══════════════════════════════════════════════════════════════════════════════

def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def hitung_cer_wer(hypothesis: str, reference: str) -> dict:
    hyp = normalize_text(hypothesis)
    ref = normalize_text(reference)

    if not ref:
        return {"cer": 0.0, "wer": 0.0}

    cer_val = round(cer(ref, hyp) * 100, 2)   # dalam persen
    wer_val = round(wer(ref, hyp) * 100, 2)   # dalam persen

    return {"cer": cer_val, "wer": wer_val}

# ══════════════════════════════════════════════════════════════════════════════
# 6. FUNGSI UTAMA EVALUASI
# ══════════════════════════════════════════════════════════════════════════════

def load_ground_truth(nama_file: str) -> str:
    """Muat file ground truth yang namanya sama dengan file gambar."""
    base = os.path.splitext(nama_file)[0]
    gt_path = os.path.join(DIR_GT, base + ".txt")
    if not os.path.exists(gt_path):
        print(f"  [PERINGATAN] Ground truth tidak ditemukan: {gt_path}")
        return ""
    with open(gt_path, "r", encoding="utf-8") as f:
        return f.read()


def bersihkan_output_lama(file_gambar_aktif: list[str]):
    if not os.path.exists(DIR_EVAL):
        return

    nama_aktif = {os.path.splitext(f)[0] for f in file_gambar_aktif}

    for folder in os.listdir(DIR_EVAL):
        folder_path = os.path.join(DIR_EVAL, folder)
        if os.path.isdir(folder_path) and folder not in nama_aktif:
            shutil.rmtree(folder_path)
            print(f"  [HAPUS] Folder lama dihapus: {folder_path}")

    if os.path.exists(CSV_OUTPUT):
        os.remove(CSV_OUTPUT)
        print(f"  [HAPUS] CSV lama dihapus: {CSV_OUTPUT}")


def run_evaluasi():
    """
    Fungsi utama: evaluasi semua gambar dengan semua teknik.
    Hasilnya disimpan ke CSV.
    """
    os.makedirs(DIR_OUTPUT, exist_ok=True)
    os.makedirs(DIR_EVAL, exist_ok=True)

    # Ambil file gambar
    ekstensi_valid = (".png",)
    file_gambar = sorted([
        f for f in os.listdir(DIR_GAMBAR)
        if f.lower().endswith(ekstensi_valid)
    ])

    if not file_gambar:
        print(f"[ERROR] Tidak ada gambar di folder: {DIR_GAMBAR}")
        return

    # Bersihkan output lama yang sudah tidak relevan
    print("[INFO] Memeriksa dan membersihkan output lama...")
    bersihkan_output_lama(file_gambar)

    print(f"[INFO] Ditemukan {len(file_gambar)} gambar")
    print(f"[INFO] Teknik yang diuji: {len(TEKNIK_LIST)}")
    print(f"[INFO] Total pengujian: {len(file_gambar) * len(TEKNIK_LIST)}\n")

    # Header CSV
    csv_header = [
        "nama_file",
        "teknik",
        "cer_persen",
        "wer_persen",
        "jumlah_karakter_ocr",
        "jumlah_karakter_gt",
        "jumlah_kata_ocr",
        "jumlah_kata_gt",
        "waktu_detik",
    ]

    semua_hasil = []

    for nama_file in file_gambar:
        path_gambar = os.path.join(DIR_GAMBAR, nama_file)
        ground_truth = load_ground_truth(nama_file)

        # Hitung jumlah kata dan karakter ground truth
        gt_word_count = len(normalize_text(ground_truth).split()) if ground_truth else 0
        gt_char_count = len(normalize_text(ground_truth).replace(" ", "")) if ground_truth else 0

        # Buat subfolder output per gambar
        base_name = os.path.splitext(nama_file)[0]
        folder_gambar = os.path.join(DIR_EVAL, base_name)
        os.makedirs(folder_gambar, exist_ok=True)

        print(f"{'='*55}")
        print(f"[GAMBAR] {nama_file}  (GT: {gt_word_count} kata)")
        print(f"{'='*55}")

        # Load gambar
        image_bgr = cv2.imread(path_gambar)
        if image_bgr is None:
            print(f"  [ERROR] Gagal membaca gambar: {path_gambar}")
            continue
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        for teknik in TEKNIK_LIST:
            print(f"  Teknik: {teknik:<15}", end=" ")
            mulai = time.time()

            try:
                # Preprocessing
                processed = process_single_technique(image, teknik)

                # OCR
                ocr_result = run_ocr(processed)

                # Hitung jumlah karakter OCR
                ocr_char_count = len(normalize_text(ocr_result["text"]).replace(" ", ""))

                # CER & WER
                if ground_truth:
                    metrik = hitung_cer_wer(ocr_result["text"], ground_truth)
                else:
                    metrik = {"cer": -1.0, "wer": -1.0}

                elapsed = round(time.time() - mulai, 2)

                # Simpan gambar hasil preprocessing
                img_out_path = os.path.join(folder_gambar, f"{teknik}_image.png")
                Image.fromarray(processed).save(img_out_path)

                # Simpan teks hasil OCR
                txt_out_path = os.path.join(folder_gambar, f"{teknik}_result.txt")
                with open(txt_out_path, "w", encoding="utf-8") as f:
                    f.write(ocr_result["text"])

                baris = {
                    "nama_file"          : nama_file,
                    "teknik"             : teknik,
                    "cer_persen"         : metrik["cer"],
                    "wer_persen"         : metrik["wer"],
                    "jumlah_karakter_ocr": ocr_char_count,
                    "jumlah_karakter_gt" : gt_char_count,
                    "jumlah_kata_ocr"    : ocr_result["word_count"],
                    "jumlah_kata_gt"     : gt_word_count,
                    "waktu_detik"        : elapsed,
                }
                semua_hasil.append(baris)

                print(
                    f"CER: {metrik['cer']:6.2f}%  "
                    f"WER: {metrik['wer']:6.2f}%  "
                    f"Waktu: {elapsed}s"
                )

            except Exception as e:
                elapsed = round(time.time() - mulai, 2)
                print(f"ERROR: {e}")
                semua_hasil.append({
                    "nama_file"          : nama_file,
                    "teknik"             : teknik,
                    "cer_persen"         : -1.0,
                    "wer_persen"         : -1.0,
                    "jumlah_karakter_ocr": 0,
                    "jumlah_karakter_gt" : gt_char_count,
                    "jumlah_kata_ocr"    : 0,
                    "jumlah_kata_gt"     : gt_word_count,
                    "waktu_detik"        : elapsed,
                })

        print()

    # ── Tulis CSV ──────────────────────────────────────────────────────────────
    with open(CSV_OUTPUT, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=csv_header)
        writer.writeheader()
        writer.writerows(semua_hasil)

    # ── Ringkasan di terminal ──────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"[SELESAI] Hasil disimpan di: {CSV_OUTPUT}")
    print(f"[SELESAI] Gambar hasil preprocessing: {DIR_EVAL}/")
    print(f"{'='*55}\n")

    print(f"{'TEKNIK':<15} {'Rata-rata CER':>14} {'Rata-rata WER':>14}")
    print("-" * 45)

    from collections import defaultdict
    cer_per_teknik = defaultdict(list)
    wer_per_teknik = defaultdict(list)

    for baris in semua_hasil:
        if baris["cer_persen"] >= 0:
            cer_per_teknik[baris["teknik"]].append(baris["cer_persen"])
            wer_per_teknik[baris["teknik"]].append(baris["wer_persen"])

    for teknik in TEKNIK_LIST:
        if cer_per_teknik[teknik]:
            avg_cer = round(sum(cer_per_teknik[teknik]) / len(cer_per_teknik[teknik]), 2)
            avg_wer = round(sum(wer_per_teknik[teknik]) / len(wer_per_teknik[teknik]), 2)
            print(f"{teknik:<15} {avg_cer:>13.2f}%  {avg_wer:>13.2f}%")

    print()


# ══════════════════════════════════════════════════════════════════════════════
# 7. ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Cek folder yang dibutuhkan
    for folder in [DIR_GAMBAR, DIR_GT]:
        if not os.path.exists(folder):
            print(f"[ERROR] Folder tidak ditemukan: {folder}")
            print(f"[INFO]  Buat folder '{folder}' dan isi dengan file yang sesuai.")
            exit(1)

    run_evaluasi()