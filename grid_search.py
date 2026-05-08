"""
ocr_grid_search.py
==================
Grid search parameter terbaik untuk preprocessing citra OCR (adaptif).
- Setiap teknik hanya aktif jika benar-benar membantu (ada guard condition).
- Hasil disimpan ke CSV secara inkremental — aman jika crash.
- Resume otomatis: iterasi yang sudah selesai akan di-skip.
- Log progress ke konsol dengan estimasi waktu selesai.

Parameter baru yang ditambahkan:
  - denoise_method    : pilih antara 'median' atau 'gaussian'
  - gaussian_kernel   : ukuran kernel jika denoise_method='gaussian'
  - binarize_method   : pilih antara 'gaussian_c' atau 'mean_c'
  - line_ratio_thresh : threshold rasio garis (sebelumnya hardcoded 0.005)
  - line_dilation_iter: jumlah iterasi dilasi masker garis
"""

import os
import re
import csv
import time
import itertools
import logging

import cv2
import numpy as np
import pytesseract
from PIL import Image
from jiwer import cer, wer

# ===========================================================
# KONFIGURASI
# ===========================================================

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
CONFIG_TESSERACT = "--oem 3 --psm 6 -l ind+eng"

DIR_GAMBAR = "data/gambar"
DIR_GT     = "data/ground_truth"
DIR_OUTPUT = "output/kombinasi"
CSV_OUTPUT = os.path.join(DIR_OUTPUT, "hasil_grid_search.csv")

os.makedirs(DIR_OUTPUT, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(DIR_OUTPUT, "grid_search.log"), mode="a", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

# ===========================================================
# PARAMETER GRID
# Kurangi isi list untuk mempercepat, tambah sesuai kebutuhan.
# ===========================================================

GRID = {
    # ── DENOISING ──────────────────────────────────────────
    # Guard: skip jika noise_level <= denoise_threshold
    "denoise_threshold": [5, 10, 15, 20,25],

    # Metode blur untuk denoising: 'median' atau 'gaussian'
    "denoise_method": ["median", "gaussian"],

    # Kernel median blur (hanya dipakai jika denoise_method='median')
    "median_kernel": [3, 5],

    # Kernel gaussian blur (hanya dipakai jika denoise_method='gaussian')
    # Harus ganjil
    "gaussian_kernel": [(3, 3), (5, 5)],

    # ── CLAHE ──────────────────────────────────────────────
    # Guard: skip jika std_intensitas >= clahe_trigger
    "clahe_trigger": [30,35, 40,45, 50],

    # clipLimit CLAHE
    "clahe_cliplimit": [1.5, 2.0, 2.5],

    # tileGridSize CLAHE
    "clahe_tile": [(4, 4), (8, 8), (16, 16)],

    # ── BINARISASI ─────────────────────────────────────────
    # Kernel median blur sebelum threshold
    "binarize_median": [3, 5],

    # blockSize adaptive threshold (harus ganjil)
    "binarize_blocksize": [31,41, 51,61, 71,81, 91,101, 111],

    # Konstanta C adaptive threshold
    "binarize_c": [25, 30,35, 40,45, 50, 55, 60],

    # Metode adaptive threshold: 'gaussian_c' atau 'mean_c'
    "binarize_method": ["gaussian_c", "mean_c"],

    # ── LINE REMOVAL ───────────────────────────────────────
    # Guard: skip jika rasio piksel garis <= threshold
    "line_ratio_thresh": [0.001, 0.003, 0.005],

    # (min_length_px, divisor_lebar) untuk panjang minimum kernel garis
    "line_removal": [(10, 40), (10, 50)],

    # Ukuran kernel dilasi masker garis
    "line_dilation": [(2, 2), (3, 3), (5, 5)],

    # Jumlah iterasi dilasi masker garis
    "line_dilation_iter": [1, 2],
}

# ===========================================================
# UTILITY
# ===========================================================

def to_grayscale(img: np.ndarray) -> np.ndarray:
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img.copy()


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def hitung_cer_wer(hypothesis: str, reference: str) -> dict:
    if not reference.strip():
        return {"cer": 0.0, "wer": 0.0}
    h = normalize_text(hypothesis)
    r = normalize_text(reference)
    return {
        "cer": round(cer(r, h) * 100, 2),
        "wer": round(wer(r, h) * 100, 2),
    }


def run_ocr(img: np.ndarray) -> str:
    pil_img = Image.fromarray(img)
    return pytesseract.image_to_string(pil_img, config=CONFIG_TESSERACT).strip()


def load_data() -> list[dict]:
    files = sorted(f for f in os.listdir(DIR_GAMBAR) if f.endswith(".png"))
    data  = []
    for f in files:
        path_img = os.path.join(DIR_GAMBAR, f)
        path_gt  = os.path.join(DIR_GT, f.replace(".png", ".txt"))

        img_bgr = cv2.imread(path_img)
        if img_bgr is None:
            log.warning(f"Gagal load gambar: {f}")
            continue

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        gt = open(path_gt, encoding="utf-8").read() if os.path.exists(path_gt) else ""
        data.append({"nama": f, "image": img_rgb, "gt": gt})

    log.info(f"Loaded {len(data)} gambar")
    return data


# ===========================================================
# PIPELINE — 1 set parameter → 1 gambar → 1 citra output
# ===========================================================

_BINARIZE_METHOD_MAP = {
    "gaussian_c": cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    "mean_c":     cv2.ADAPTIVE_THRESH_MEAN_C,
}


def preprocess(
    image: np.ndarray,
    # denoising
    denoise_threshold: int,
    denoise_method: str,
    median_kernel: int,
    gaussian_kernel: tuple,
    # clahe
    clahe_trigger: float,
    clahe_cliplimit: float,
    clahe_tile: tuple,
    # binarisasi
    binarize_median: int,
    binarize_blocksize: int,
    binarize_c: int,
    binarize_method: str,
    # line removal
    line_ratio_thresh: float,
    line_removal: tuple,
    line_dilation: tuple,
    line_dilation_iter: int,
) -> np.ndarray:

    # ── Grayscale ─────────────────────────────────────────
    gray = to_grayscale(image)

    # ── TAHAP 1: DENOISING (adaptif) ──────────────────────
    blurred_ref = cv2.GaussianBlur(gray, (5, 5), 0)
    noise_level = float(np.std(gray.astype(np.float32) - blurred_ref.astype(np.float32)))

    if noise_level > denoise_threshold:
        if denoise_method == "median":
            img_dn = cv2.medianBlur(gray, median_kernel)
        else:  # gaussian
            img_dn = cv2.GaussianBlur(gray, gaussian_kernel, 0)
    else:
        img_dn = gray  # citra cukup bersih → skip

    # ── TAHAP 2: CLAHE (adaptif) ──────────────────────────
    std_val = float(np.std(img_dn))

    if std_val < clahe_trigger:
        enhancer = cv2.createCLAHE(clipLimit=clahe_cliplimit, tileGridSize=clahe_tile)
        img_cl   = enhancer.apply(img_dn)
    else:
        img_cl = img_dn  # kontras sudah cukup → skip

    # ── TAHAP 3: BINARISASI ADAPTIF ───────────────────────
    img_blur = cv2.medianBlur(img_cl, binarize_median)

    # Pastikan blockSize selalu ganjil
    bs = binarize_blocksize if binarize_blocksize % 2 == 1 else binarize_blocksize + 1

    thresh_method = _BINARIZE_METHOD_MAP.get(binarize_method, cv2.ADAPTIVE_THRESH_GAUSSIAN_C)

    img_bin = cv2.adaptiveThreshold(
        img_blur, 255,
        thresh_method,
        cv2.THRESH_BINARY,
        bs, binarize_c,
    )

    # ── TAHAP 4: PENGHAPUSAN GARIS (adaptif) ──────────────
    lmin, ldiv = line_removal
    h, w       = img_bin.shape
    inv        = cv2.bitwise_not(img_bin)

    kh = cv2.getStructuringElement(cv2.MORPH_RECT, (max(lmin, w // ldiv), 1))
    kv = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(lmin, h // ldiv)))

    line_h = cv2.morphologyEx(inv, cv2.MORPH_OPEN, kh)
    line_v = cv2.morphologyEx(inv, cv2.MORPH_OPEN, kv)
    mask   = cv2.add(line_h, line_v)

    line_ratio = np.count_nonzero(mask) / (h * w)

    if line_ratio > line_ratio_thresh:
        mask_dil = cv2.dilate(
            mask,
            np.ones(line_dilation, np.uint8),
            iterations=line_dilation_iter,
        )
        cleaned = cv2.bitwise_and(inv, inv, mask=cv2.bitwise_not(mask_dil))
        final   = cv2.bitwise_not(cleaned)
    else:
        final = img_bin  # tidak ada garis signifikan → skip

    return final


# ===========================================================
# RESUME: baca kombinasi yang sudah selesai dari CSV
# ===========================================================

CSV_COLUMNS = [
    # denoising
    "denoise_threshold", "denoise_method", "median_kernel", "gaussian_kernel",
    # clahe
    "clahe_trigger", "clahe_cliplimit", "clahe_tile",
    # binarisasi
    "binarize_median", "binarize_blocksize", "binarize_c", "binarize_method",
    # line removal
    "line_ratio_thresh", "line_removal", "line_dilation", "line_dilation_iter",
    # hasil
    "avg_cer", "avg_wer",
]

PARAM_KEYS = [k for k in CSV_COLUMNS if k not in ("avg_cer", "avg_wer")]


def load_done_keys(csv_path: str) -> set:
    done = set()
    if not os.path.exists(csv_path):
        return done
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            done.add(tuple(row[k] for k in PARAM_KEYS))
    return done


def param_to_key(p: dict) -> tuple:
    return tuple(str(p[k]) for k in PARAM_KEYS)


# ===========================================================
# MAIN
# ===========================================================

def run():
    data = load_data()
    if not data:
        log.error("Tidak ada data ditemukan. Hentikan.")
        return

    keys             = list(GRID.keys())
    all_combinations = list(itertools.product(*GRID.values()))
    total            = len(all_combinations)
    log.info(f"Total kombinasi: {total:,}")

    done_keys = load_done_keys(CSV_OUTPUT)
    log.info(f"Kombinasi sudah selesai (di-skip): {len(done_keys):,}")

    csv_file_exists = os.path.exists(CSV_OUTPUT)
    csv_fp  = open(CSV_OUTPUT, "a", newline="", encoding="utf-8")
    writer  = csv.DictWriter(csv_fp, fieldnames=CSV_COLUMNS)
    if not csv_file_exists:
        writer.writeheader()
        csv_fp.flush()

    # Baca best dari run sebelumnya
    best_cer   = 999.0
    best_param = None
    if done_keys:
        with open(CSV_OUTPUT, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                val = float(row["avg_cer"])
                if val < best_cer:
                    best_cer   = val
                    best_param = dict(row)

    iter_count = 0
    skip_count = 0
    t_start    = time.time()

    for combo in all_combinations:
        params = dict(zip(keys, combo))
        pkey   = param_to_key(params)

        if pkey in done_keys:
            skip_count += 1
            continue

        iter_count += 1
        global_idx  = skip_count + iter_count

        elapsed   = time.time() - t_start
        rate      = iter_count / elapsed if elapsed > 0 else 0
        remaining = (total - global_idx) / rate if rate > 0 else float("inf")
        eta_str   = f"{remaining/60:.1f} mnt" if remaining < 3600 else f"{remaining/3600:.1f} jam"

        log.info(
            f"[{global_idx}/{total}] "
            f"dn={params['denoise_threshold']}({params['denoise_method']}) "
            f"tr={params['clahe_trigger']} cl={params['clahe_cliplimit']} "
            f"bs={params['binarize_blocksize']} c={params['binarize_c']} "
            f"bm={params['binarize_method']} | ETA ~{eta_str}"
        )

        cer_list = []
        wer_list = []

        for d in data:
            try:
                img_out = preprocess(d["image"], **params)
                text    = run_ocr(img_out)
                metrics = hitung_cer_wer(text, d["gt"])
                cer_list.append(metrics["cer"])
                wer_list.append(metrics["wer"])
            except Exception as e:
                log.warning(f"  Error pada {d['nama']}: {e}")
                cer_list.append(999.0)
                wer_list.append(999.0)

        avg_cer = float(np.mean(cer_list)) if cer_list else 999.0
        avg_wer = float(np.mean(wer_list)) if wer_list else 999.0

        log.info(f"  → avg CER={avg_cer:.2f}%  avg WER={avg_wer:.2f}%")

        row = {k: str(params[k]) for k in PARAM_KEYS}
        row["avg_cer"] = round(avg_cer, 4)
        row["avg_wer"] = round(avg_wer, 4)
        writer.writerow(row)
        csv_fp.flush()

        if avg_cer < best_cer:
            best_cer   = avg_cer
            best_param = row
            log.info(f"  ★ NEW BEST CER={best_cer:.2f}%")

    csv_fp.close()

    log.info("\n" + "=" * 60)
    log.info(f"SELESAI — {iter_count} kombinasi baru diproses")
    log.info(f"BEST CER  : {best_cer:.2f}%")
    log.info(f"BEST PARAM: {best_param}")
    log.info(f"Hasil lengkap: {CSV_OUTPUT}")


if __name__ == "__main__":
    for folder in [DIR_GAMBAR, DIR_GT]:
        if not os.path.exists(folder):
            print(f"[ERROR] Folder tidak ditemukan: {folder}")
            exit(1)
    run()