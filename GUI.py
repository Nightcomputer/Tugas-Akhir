import os, re, cv2, csv, io, time, queue, threading
import numpy as np
import pytesseract
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
from collections import defaultdict

try:
    from jiwer import cer as _cer, wer as _wer
    JIWER_OK = True
except ImportError:
    JIWER_OK = False

try:
    import fitz
    FITZ_OK = True
except ImportError:
    FITZ_OK = False

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
CONFIG_TESSERACT = "--oem 3 --psm 6 -l ind+eng"

# Parameter teknik preprocessing — dipakai bersama oleh pipeline ASLI (di atas)
# maupun oleh perhitungan visualisasi/edukasi (kelas VisualisasiFrame di bawah),
# supaya keduanya selalu konsisten walau nilainya diubah di kemudian hari.
CLAHE_CLIP_LIMIT     = 2.5
CLAHE_TILE_GRID      = (4, 4)      # (kolom, baris) — urutan tileGridSize OpenCV: (lebar, tinggi)
CLAHE_STD_THRESHOLD  = 30.0
AGT_BLOCK_SIZE       = 111
AGT_C                = 55
LINE_RATIO_THRESHOLD = 0.003

BG      = "#0d1117"
PANEL   = "#161b22"
CARD    = "#1c2130"
BORDER  = "#30363d"
AKSEN   = "#4f87f7"
AKSEN2  = "#7c5aed"
HIJAU   = "#2ea043"
MERAH   = "#da3633"
KUNING  = "#d29922"
TEAL    = "#39c5cf"
TEKS    = "#e6edf3"
TEKS2   = "#7d8590"
TEKS3   = "#444c56"
PUTIH   = "#ffffff"
TERM_BG = "#0a0d14"
NAV_BG  = "#010409"

F_JUDUL = ("Segoe UI", 10, "bold")
F_LABEL = ("Segoe UI",  9)
F_KECIL = ("Segoe UI",  8)
F_MONO  = ("Consolas",  9)
F_BADGE = ("Segoe UI",  7, "bold")
F_TERM  = ("Consolas",  9)

EVAL_TEKNIK = [
    "none",
    "grayscale",
    "clahe",
    "adaptive_gaussian_thresholding",
    "line_removal",
    "adaptif preprocessing",
]


def to_grayscale(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return image.copy()


def clahe_adaptive(gray: np.ndarray):
    std = float(np.std(gray))
    if std >= CLAHE_STD_THRESHOLD:
        return gray, False
    enh = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID)
    return enh.apply(gray), True


def clahe_paksa(gray: np.ndarray) -> np.ndarray:
    enh = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID)
    return enh.apply(gray)


def binarize(gray: np.ndarray) -> np.ndarray:
    gray = cv2.medianBlur(gray, 3)
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, AGT_BLOCK_SIZE, AGT_C)


def remove_lines_adaptive(binary: np.ndarray):
    h, w  = binary.shape
    inv   = ~binary
    kh    = cv2.getStructuringElement(cv2.MORPH_RECT, (max(10, w // 40), 1))
    kv    = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(10, h // 40)))
    mask  = cv2.add(cv2.morphologyEx(inv, cv2.MORPH_OPEN, kh),
                    cv2.morphologyEx(inv, cv2.MORPH_OPEN, kv))
    if np.count_nonzero(mask) / (h * w) <= LINE_RATIO_THRESHOLD:
        return binary, False
    mask  = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
    hasil = cv2.bitwise_and(inv, inv, mask=cv2.bitwise_not(mask))
    return ~hasil, True


def remove_lines_paksa(binary: np.ndarray) -> np.ndarray:
    h, w  = binary.shape
    inv   = ~binary
    kh    = cv2.getStructuringElement(cv2.MORPH_RECT, (max(10, w // 40), 1))
    kv    = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(10, h // 40)))
    mask  = cv2.add(cv2.morphologyEx(inv, cv2.MORPH_OPEN, kh),
                    cv2.morphologyEx(inv, cv2.MORPH_OPEN, kv))
    mask  = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
    hasil = cv2.bitwise_and(inv, inv, mask=cv2.bitwise_not(mask))
    return ~hasil


def pipeline_adaptive(image: np.ndarray):
    log = {}
    img = to_grayscale(image)
    log["grayscale"] = True
    img, log["clahe"] = clahe_adaptive(img)
    img = binarize(img)
    log["binarize"] = True
    img, log["line_removal"] = remove_lines_adaptive(img)
    return img, log


def pipeline_manual(image: np.ndarray, aktif: dict):
    if not aktif.get("Grayscale", True):
        return image.copy(), image.copy()
    img = to_grayscale(image)
    if aktif.get("CLAHE", False):
        img = clahe_paksa(img)
    if aktif.get("Adaptive Gaussian", True):
        img = binarize(img)
    if aktif.get("Line Removal", False):
        img = remove_lines_paksa(img)
    return img, img


def process_single_technique(image: np.ndarray, technique: str) -> np.ndarray:
    if technique == "none":
        return image.copy()
    elif technique == "grayscale":
        return to_grayscale(image)
    elif technique == "clahe":
        return clahe_paksa(to_grayscale(image))
    elif technique == "adaptive_gaussian_thresholding":
        return binarize(to_grayscale(image))
    elif technique == "line_removal":
        img, _ = remove_lines_adaptive(binarize(to_grayscale(image)))
        return img
    elif technique == "adaptif preprocessing":
        img, _ = pipeline_adaptive(image)
        return img
    else:
        raise ValueError(f"Teknik tidak dikenal: '{technique}'")


def run_ocr(image: np.ndarray) -> str:
    pil = Image.fromarray(image)
    t   = pytesseract.image_to_string(pil, config=CONFIG_TESSERACT)
    t   = t.replace("|", "")
    t   = re.sub(r'\n\s*\n+', '\n', t)
    return "\n".join(line.strip() for line in t.splitlines()).strip()


def normalize_text(text: str) -> str:
    return re.sub(r'\s+', ' ', text.lower()).strip()


def hitung_cer_wer(hyp: str, ref: str) -> dict:
    if not ref.strip() or not JIWER_OK:
        return {"cer": None, "wer": None}
    try:
        h, r = normalize_text(hyp), normalize_text(ref)
        return {"cer": round(_cer(r, h) * 100, 2),
                "wer": round(_wer(r, h) * 100, 2)}
    except Exception:
        return {"cer": None, "wer": None}


def analisis_gambar(img_gray: np.ndarray) -> dict:
    intensitas = round(float(np.mean(img_gray)), 2)
    if len(img_gray.shape) == 3:
        img_gray = cv2.cvtColor(img_gray, cv2.COLOR_RGB2GRAY)
    if img_gray.dtype != np.uint8:
        img_gray = np.clip(img_gray, 0, 255).astype(np.uint8)
    temp     = cv2.medianBlur(img_gray, 3)
    temp_bin = cv2.adaptiveThreshold(temp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 111, 55)
    h, w  = temp_bin.shape
    inv   = cv2.bitwise_not(temp_bin)
    kh    = cv2.getStructuringElement(cv2.MORPH_RECT, (max(10, w // 40), 1))
    kv    = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(10, h // 40)))
    mask  = cv2.add(cv2.morphologyEx(inv, cv2.MORPH_OPEN, kh),
                    cv2.morphologyEx(inv, cv2.MORPH_OPEN, kv))
    rasio = round(np.count_nonzero(mask) / (h * w) * 100, 3)
    return {"intensitas": intensitas, "rasio_garis": rasio, "ada_garis": rasio > 0.3}


# =====================================================================
#  FUNGSI PERHITUNGAN UNTUK MENU "VISUALISASI PROSES"
#  Semua fungsi di bawah ini murni untuk menampilkan/menjelaskan CARA
#  setiap teknik bekerja langkah demi langkah. Pipeline asli di atas
#  (clahe_adaptive, binarize, dst) TIDAK diubah dan TIDAK memanggil
#  fungsi-fungsi ini.
# =====================================================================

def viz_grayscale_sample(image_rgb: np.ndarray, titik=None, ukuran: int = 4) -> dict:
    """Ambil patch kecil ukuran x ukuran piksel lalu hitung konversi
    grayscale-nya satu per satu:
        I(x,y) = 0.299*R(x,y) + 0.587*G(x,y) + 0.114*B(x,y)
    """
    h, w = image_rgb.shape[:2]
    if titik is None:
        cy, cx = h // 2, w // 2
    else:
        cy, cx = titik
    cy = max(0, min(h - 1, cy))
    cx = max(0, min(w - 1, cx))
    y0 = max(0, min(h - ukuran, cy - ukuran // 2))
    x0 = max(0, min(w - ukuran, cx - ukuran // 2))
    patch = image_rgb[y0:y0 + ukuran, x0:x0 + ukuran].astype(np.float64)

    contoh = []
    for i in range(ukuran):
        for j in range(ukuran):
            r, g, b = patch[i, j]
            exact = 0.299 * r + 0.587 * g + 0.114 * b
            contoh.append({
                "y": y0 + i, "x": x0 + j,
                "r": int(r), "g": int(g), "b": int(b),
                "gray_exact": exact, "gray_bulat": int(round(exact)),
            })
    return {
        "y0": y0, "x0": x0, "ukuran": ukuran, "titik": (cy, cx),
        "contoh": contoh,
        "patch_asli": image_rgb[y0:y0 + ukuran, x0:x0 + ukuran],
        "patch_gray": to_grayscale(image_rgb)[y0:y0 + ukuran, x0:x0 + ukuran],
    }


def viz_std_deviasi(gray: np.ndarray, threshold: float = CLAHE_STD_THRESHOLD) -> dict:
    """Hitung standar deviasi intensitas seluruh citra lalu tentukan
    keputusan aktivasi CLAHE (aktif bila std < threshold).
    """
    gray_f = gray.astype(np.float64)
    n = int(gray.size)
    jumlah = float(np.sum(gray_f))
    mean = jumlah / n
    jumlah_sqdiff = float(np.sum((gray_f - mean) ** 2))
    variansi = jumlah_sqdiff / n
    std = float(np.sqrt(variansi))
    return {
        "n": n, "jumlah": jumlah, "mean": mean,
        "jumlah_sqdiff": jumlah_sqdiff, "variansi": variansi, "std": std,
        "threshold": threshold, "aktif": std < threshold,
    }


def viz_bagi_tile(gray: np.ndarray, grid=CLAHE_TILE_GRID) -> list:
    """Bagi citra grayscale menjadi grid tile seperti yang dipakai tileGridSize
    pada cv2.createCLAHE. Mengembalikan list 2D (per baris, per kolom) berisi
    info batas & potongan citra tiap tile.

    PENTING: tileGridSize OpenCV memakai urutan (kolom, baris) — yaitu
    (jumlah tile arah lebar/x, jumlah tile arah tinggi/y) — BUKAN (baris,
    kolom). Sudah diverifikasi empiris: cv2.createCLAHE(tileGridSize=(8,1))
    membagi LEBAR jadi 8 (bukan tinggi), dan tileGridSize=(1,8) membagi
    TINGGI jadi 8. Kalau grid persegi seperti (4,4) urutan ini tidak
    kelihatan bedanya, tapi harus tetap benar supaya kalau CLAHE_TILE_GRID
    diganti ke nilai tidak simetris (mis. (2,4)), tile yang divisualisasikan
    di sini tetap sama persis dengan yang dipakai pipeline asli.
    """
    h, w = gray.shape
    kolom, baris = grid
    y_bounds = [round(i * h / baris) for i in range(baris + 1)]
    x_bounds = [round(j * w / kolom) for j in range(kolom + 1)]
    tiles = []
    for i in range(baris):
        row = []
        for j in range(kolom):
            y0, y1 = y_bounds[i], y_bounds[i + 1]
            x0, x1 = x_bounds[j], x_bounds[j + 1]
            row.append({
                "baris": i, "kolom": j, "y0": y0, "y1": y1, "x0": x0, "x1": x1,
                "citra": gray[y0:y1, x0:x1],
            })
        tiles.append(row)
    return tiles


def viz_histogram_tile_manual(tile_img: np.ndarray,
                               clip_limit: float = CLAHE_CLIP_LIMIT, L: int = 256) -> dict:
    """Hitung histogram, clipping, redistribusi, dan pemetaan intensitas
    SATU tile:
        H_clip(k)  = min(H(k), Tc)
        kelebihan  = jumlah(H(k) - H_clip(k)) untuk semua k   (piksel yang terpotong)
        H_final(k) = H_clip(k) + kelebihan / L                (disebar rata ke semua level)
        s_k        = (L-1) * cumsum(H_final(j)) / N_tile

    Tahap redistribusi ini penting: tanpanya, tile dengan jumlah piksel
    besar (mis. hasil scan dokumen penuh) akan selalu menghasilkan s_k
    yang sangat kecil di semua level (karena total piksel yang lolos
    clipping dibatasi oleh L x Tc, angka tetap, sementara N_tile jauh
    lebih besar) — sehingga tile tampak hitam. Dengan redistribusi,
    seluruh piksel yang terpotong dikembalikan, jadi s_k selalu mencapai
    (L-1) tepat di level tertinggi, persis seperti perhitungan histogram
    equalization standar dengan batasan kontras.
    """
    n_tile = int(tile_img.size)
    hist = np.bincount(tile_img.flatten(), minlength=L).astype(np.float64)
    hist_clip = np.minimum(hist, clip_limit)
    kelebihan = float((hist - hist_clip).sum())
    hist_final = hist_clip + (kelebihan / L)
    kumulatif = np.cumsum(hist_final)
    s_k = (L - 1) * kumulatif / n_tile
    lut = np.clip(np.round(s_k), 0, 255).astype(np.uint8)
    hasil = lut[tile_img]
    return {
        "n_tile": n_tile, "L": L, "Tc": clip_limit,
        "hist": hist, "hist_clip": hist_clip, "hist_final": hist_final, "kumulatif": kumulatif,
        "s_k": s_k, "lut": lut, "hasil": hasil,
        "sum_hist": float(hist.sum()), "sum_hist_clip": float(hist_clip.sum()),
    }


def viz_titik_agt(gray: np.ndarray, titik=None,
                   block_size: int = AGT_BLOCK_SIZE, C: int = AGT_C) -> dict:
    """Hitung threshold adaptif Gaussian pada SATU titik memakai window
    penuh (block_size x block_size), dengan bobot Gaussian & nilai sigma
    yang identik dengan yang dipakai cv2.adaptiveThreshold
    (ADAPTIVE_THRESH_GAUSSIAN_C) secara internal.
    """
    h, w = gray.shape
    if titik is None:
        cy, cx = h // 2, w // 2
    else:
        cy, cx = titik
    cy = max(0, min(h - 1, cy))
    cx = max(0, min(w - 1, cx))
    r = block_size // 2
    blur = cv2.medianBlur(gray, 3)

    y0, x0 = cy - r, cx - r
    y1, x1 = cy + r + 1, cx + r + 1
    pad_top, pad_bottom = max(0, -y0), max(0, y1 - h)
    pad_left, pad_right = max(0, -x0), max(0, x1 - w)
    blur_pad = cv2.copyMakeBorder(blur, pad_top, pad_bottom, pad_left, pad_right,
                                   cv2.BORDER_REPLICATE)
    window = blur_pad[y0 + pad_top:y0 + pad_top + block_size,
                       x0 + pad_left:x0 + pad_left + block_size].astype(np.float64)

    kernel_1d = cv2.getGaussianKernel(block_size, 0)
    sigma = 0.3 * ((block_size - 1) * 0.5 - 1) + 0.8
    bobot = kernel_1d @ kernel_1d.T

    sum_bobot = float(np.sum(bobot))
    sum_bobot_i = float(np.sum(bobot * window))
    T = sum_bobot_i / sum_bobot - C
    pusat_blur = float(blur[cy, cx])
    hasil = 255 if pusat_blur > T else 0

    rr = min(r, cy, cx, h - 1 - cy, w - 1 - cx)
    return {
        "titik": (cy, cx), "block_size": block_size, "C": C, "sigma": sigma,
        "sum_bobot": sum_bobot, "sum_bobot_i": sum_bobot_i, "T": T,
        "pusat_blur": pusat_blur, "pusat_asli": float(gray[cy, cx]), "hasil": hasil,
        "window_asli_crop": gray[cy - rr:cy + rr + 1, cx - rr:cx + rr + 1],
        "window_blur_crop": blur[cy - rr:cy + rr + 1, cx - rr:cx + rr + 1],
    }


def viz_line_removal_detail(binary: np.ndarray, threshold: float = LINE_RATIO_THRESHOLD) -> dict:
    """Jalankan & tangkap SETIAP tahap penghapusan garis tabel: biner ->
    invers -> opening horizontal/vertikal -> mask gabungan -> rasio Rg
    (keputusan aktivasi) -> dilasi mask -> hasil akhir. Perhitungan ini
    identik dengan remove_lines_adaptive/_paksa di atas (dipecah per-langkah
    supaya bisa ditampilkan satu-satu).
    """
    h, w = binary.shape
    inv = ~binary
    kh = cv2.getStructuringElement(cv2.MORPH_RECT, (max(10, w // 40), 1))
    kv = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(10, h // 40)))
    buka_h = cv2.morphologyEx(inv, cv2.MORPH_OPEN, kh)
    buka_v = cv2.morphologyEx(inv, cv2.MORPH_OPEN, kv)
    mask = cv2.add(buka_h, buka_v)

    p_garis = int(np.count_nonzero(mask))
    p_total = int(h * w)
    rg = p_garis / p_total
    aktif = rg > threshold

    mask_dilasi = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
    hasil_and = cv2.bitwise_and(inv, inv, mask=cv2.bitwise_not(mask_dilasi))
    final = ~hasil_and

    return {
        "biner": binary, "inv": inv,
        "kh_size": (kh.shape[1], kh.shape[0]), "kv_size": (kv.shape[1], kv.shape[0]),
        "buka_h": buka_h, "buka_v": buka_v, "mask": mask,
        "p_garis": p_garis, "p_total": p_total, "rg": rg,
        "threshold": threshold, "aktif": aktif,
        "mask_dilasi": mask_dilasi, "final": final if aktif else binary,
    }


def viz_levenshtein_detail(ref_tokens: list, hyp_tokens: list) -> dict:
    """Hitung matriks Levenshtein Distance LENGKAP antara dua urutan token
    (bisa berupa list karakter untuk CER, atau list kata untuk WER), lalu
    telusuri balik (backtrack) jalur optimalnya untuk memecah totalnya
    menjadi jumlah substitusi (Sc/Sw), penghapusan (Dc/Dw), dan penambahan
    (Ic/Iw).
    """
    m, n = len(ref_tokens), len(hyp_tokens)
    d = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        d[i][0] = i
    for j in range(n + 1):
        d[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if ref_tokens[i - 1] == hyp_tokens[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)

    i, j = m, n
    sub = hapus = tambah = 0
    jalur = []
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref_tokens[i - 1] == hyp_tokens[j - 1] and d[i][j] == d[i - 1][j - 1]:
            jalur.append((i, j, "cocok"))
            i, j = i - 1, j - 1
        elif i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + 1:
            sub += 1
            jalur.append((i, j, "substitusi"))
            i, j = i - 1, j - 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1:
            hapus += 1
            jalur.append((i, j, "hapus"))
            i -= 1
        elif j > 0 and d[i][j] == d[i][j - 1] + 1:
            tambah += 1
            jalur.append((i, j, "tambah"))
            j -= 1
        else:
            break
    jalur.reverse()
    return {
        "matriks": d, "m": m, "n": n, "ref": ref_tokens, "hyp": hyp_tokens,
        "sub": sub, "hapus": hapus, "tambah": tambah,
        "distance": d[m][n], "jalur": set((i, j) for i, j, _ in jalur),
        "jalur_jenis": {(i, j): jenis for i, j, jenis in jalur},
    }


class ZoomableCanvas(tk.Canvas):
    ZOOM_STEP = 1.25
    ZOOM_MIN  = 0.1
    ZOOM_MAX  = 20.0

    def __init__(self, parent, **kw):
        kw.setdefault("bg", TERM_BG)
        kw.setdefault("highlightthickness", 0)
        kw.setdefault("cursor", "crosshair")
        super().__init__(parent, **kw)
        self._pil_src    = None
        self._tk_img     = None
        self._zoom       = 1.0
        self._fit_scale  = 1.0
        self._offset_x   = 0.0
        self._offset_y   = 0.0
        self._drag_start = None
        self.bind("<Configure>",       self._on_resize)
        self.bind("<MouseWheel>",      self._on_scroll_win)
        self.bind("<Button-4>",        self._on_scroll_up)
        self.bind("<Button-5>",        self._on_scroll_down)
        self.bind("<ButtonPress-1>",   self._on_drag_start)
        self.bind("<B1-Motion>",       self._on_drag_move)
        self.bind("<ButtonRelease-1>", self._on_drag_end)

    def set_image(self, img: np.ndarray, gray: bool = False):
        self._pil_src = (Image.fromarray(img).convert("RGB")
                         if (gray and len(img.shape) == 2)
                         else Image.fromarray(img))
        self._reset_zoom()

    def clear(self):
        self._pil_src = self._tk_img = None
        self.delete("all")

    def reset_zoom(self):
        self._reset_zoom()

    def _reset_zoom(self):
        self.update_idletasks()
        cw, ch = self.winfo_width() or 500, self.winfo_height() or 400
        if not self._pil_src:
            return
        iw, ih = self._pil_src.size
        self._fit_scale      = min(cw / iw, ch / ih, 1.0)
        self._zoom           = 1.0
        self._offset_x, self._offset_y = cw / 2, ch / 2
        self._render()

    def _render(self):
        if not self._pil_src:
            return
        self.update_idletasks()
        sc      = self._fit_scale * self._zoom
        iw, ih  = self._pil_src.size
        nw, nh  = max(1, int(iw * sc)), max(1, int(ih * sc))
        self._tk_img = ImageTk.PhotoImage(self._pil_src.resize((nw, nh), Image.LANCZOS))
        self.delete("img")
        self.create_image(self._offset_x, self._offset_y,
                          image=self._tk_img, anchor="center", tags="img")

    def _on_resize(self, e):
        if not self._pil_src:
            return
        iw, ih = self._pil_src.size
        self._fit_scale          = min(e.width / iw, e.height / ih, 1.0)
        self._offset_x, self._offset_y = e.width / 2, e.height / 2
        self._render()

    def _zoom_at(self, x, y, f):
        nz = max(self.ZOOM_MIN, min(self.ZOOM_MAX, self._zoom * f))
        r  = nz / self._zoom
        self._offset_x = x + (self._offset_x - x) * r
        self._offset_y = y + (self._offset_y - y) * r
        self._zoom = nz
        self._render()

    def _on_scroll_win(self, e):
        self._zoom_at(e.x, e.y, self.ZOOM_STEP if e.delta > 0 else 1 / self.ZOOM_STEP)
    def _on_scroll_up(self,   e): self._zoom_at(e.x, e.y, self.ZOOM_STEP)
    def _on_scroll_down(self, e): self._zoom_at(e.x, e.y, 1 / self.ZOOM_STEP)

    def _on_drag_start(self, e):
        self._drag_start = (e.x, e.y)
        self.configure(cursor="fleur")

    def _on_drag_move(self, e):
        if not self._drag_start:
            return
        self._offset_x  += e.x - self._drag_start[0]
        self._offset_y  += e.y - self._drag_start[1]
        self._drag_start = (e.x, e.y)
        self._render()

    def _on_drag_end(self, e):
        self._drag_start = None
        self.configure(cursor="crosshair")


def buat_tombol(parent, text, cmd, warna=AKSEN, lebar=14):
    b = tk.Button(parent, text=text, command=cmd, font=F_LABEL,
                  bg=warna, fg=PUTIH, activebackground=warna, activeforeground=PUTIH,
                  relief="flat", bd=0, width=lebar, cursor="hand2", padx=8, pady=5)
    def dk(c):
        c = c.lstrip("#")
        r, g, v = int(c[:2], 16), int(c[2:4], 16), int(c[4:], 16)
        return f"#{max(0,r-25):02x}{max(0,g-25):02x}{max(0,v-25):02x}"
    b.bind("<Enter>", lambda e: b.config(bg=dk(warna)))
    b.bind("<Leave>", lambda e: b.config(bg=warna))
    return b


def judul_seksi(parent, teks, warna_aksen=AKSEN, canvas_zoom=None):
    f = tk.Frame(parent, bg=PANEL)
    f.pack(fill="x", padx=12, pady=(10, 6))
    tk.Frame(f, bg=warna_aksen, width=3, height=16).pack(side="left", padx=(0, 8))
    tk.Label(f, text=teks, font=F_JUDUL, bg=PANEL, fg=PUTIH).pack(side="left")
    if canvas_zoom:
        tk.Button(f, text="⊡ Reset Zoom", command=canvas_zoom.reset_zoom,
                  font=F_KECIL, bg="#21262d", fg=TEKS2,
                  activebackground="#30363d", activeforeground=TEKS,
                  relief="flat", bd=0, padx=6, pady=2, cursor="hand2").pack(side="right")


def _buat_terminal_panel(parent, warna_aksen=AKSEN2, judul="Output Terminal", tinggi=24):
    """Buat panel terminal yang konsisten di semua halaman."""
    hdr = tk.Frame(parent, bg=PANEL)
    hdr.pack(fill="x")
    tk.Frame(hdr, bg=PANEL, height=10).pack(fill="x")
    inner = tk.Frame(hdr, bg=PANEL)
    inner.pack(fill="x", padx=16)
    tk.Frame(inner, bg=warna_aksen, width=3, height=14).pack(side="left", padx=(0, 8))
    tk.Label(inner, text=judul, font=F_JUDUL, bg=PANEL, fg=PUTIH).pack(side="left")
    tk.Frame(hdr, bg=PANEL, height=10).pack(fill="x")
    tk.Frame(parent, bg=BORDER, height=1).pack(fill="x")

    term_f = tk.Frame(parent, bg=TERM_BG)
    term_f.pack(fill="both", expand=True)

    txt = tk.Text(term_f, font=F_TERM, bg=TERM_BG, fg=TEKS,
                  insertbackground=TEKS, relief="flat", bd=0, height=tinggi,
                  wrap="none", padx=14, pady=12, state="disabled")
    sb_y = ttk.Scrollbar(term_f, orient="vertical",   command=txt.yview)
    sb_x = ttk.Scrollbar(term_f, orient="horizontal", command=txt.xview)
    txt.configure(yscrollcommand=sb_y.set, xscrollcommand=sb_x.set)
    sb_y.pack(side="right",  fill="y")
    sb_x.pack(side="bottom", fill="x")
    txt.pack(fill="both", expand=True)

    txt.tag_configure("normal",  foreground=TEKS)
    txt.tag_configure("header",  foreground=warna_aksen, font=("Consolas", 9, "bold"))
    txt.tag_configure("aksen",   foreground=AKSEN,  font=("Consolas", 9, "bold"))
    txt.tag_configure("aksen2",  foreground=AKSEN2)
    txt.tag_configure("hijau",   foreground=HIJAU)
    txt.tag_configure("merah",   foreground=MERAH)
    txt.tag_configure("kuning",  foreground=KUNING)
    txt.tag_configure("abu",     foreground=TEKS3)
    txt.tag_configure("sukses",  foreground=HIJAU, font=("Consolas", 9, "bold"))
    txt.tag_configure("bold",    font=("Consolas", 9, "bold"))

    return txt


class AppFrame(tk.Frame):
    TEKNIK_LIST = ["Grayscale", "CLAHE", "Adaptive Gaussian", "Line Removal"]

    def __init__(self, parent, **kw):
        super().__init__(parent, bg=BG, **kw)
        self.img_asli      = None
        self.img_prep      = None
        self.teks_gt       = ""
        self.teks_ocr_asli = ""
        self.teks_ocr_prep = ""
        self.log_teknik    = {}
        self.dimatikan     = set()
        self.aktif_teknik  = {}
        self.aktif_awal    = {}
        self.mode_adaptif  = False
        self._analisis_awal = None
        self._bangun_ui()

    def _bangun_ui(self):
        self._bangun_header()
        self._bangun_scroll_container()

        self.frame_atas = tk.Frame(self._scroll_inner, bg=BG, width=800, height=500)
        self.frame_atas.pack(fill="x", expand=False, padx=12, pady=(0, 4))
        self.frame_atas.pack_propagate(False)
        self.frame_atas.columnconfigure(0, weight=4, uniform="atas", minsize=200)
        self.frame_atas.columnconfigure(1, weight=4, uniform="atas", minsize=200)
        self.frame_atas.columnconfigure(2, weight=2, uniform="atas", minsize=140)
        self.frame_atas.rowconfigure(0, weight=1)

        self._bangun_panel_gambar_asli(self.frame_atas)
        self._bangun_panel_gambar_prep(self.frame_atas)
        self._bangun_panel_kontrol_analisis(self.frame_atas)

        self.frame_bawah = tk.Frame(self._scroll_inner, bg=BG, width=800)
        self.frame_bawah.pack(fill="x", expand=False, padx=12, pady=(0, 10))
        self.frame_bawah.pack_propagate(False)
        self.frame_bawah.columnconfigure(0, weight=1, uniform="bawah", minsize=200)
        self.frame_bawah.columnconfigure(1, weight=1, uniform="bawah", minsize=200)
        self.frame_bawah.columnconfigure(2, weight=1, uniform="bawah", minsize=200)

        self._bangun_panel_gt(self.frame_bawah)
        self._bangun_panel_ocr_asli(self.frame_bawah)
        self._bangun_panel_ocr_prep(self.frame_bawah)

    def _bangun_header(self):
        hdr = tk.Frame(self, bg=PANEL, height=58)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)

        kiri = tk.Frame(hdr, bg=PANEL)
        kiri.pack(side="left", padx=20, pady=10)
        tk.Frame(kiri, bg=AKSEN, width=4, height=30).pack(side="left", padx=(0, 10))
        tk.Label(kiri, text="Sistem Preprocessing Adaptif OCR",
                 font=("Segoe UI", 13, "bold"), bg=PANEL, fg=PUTIH).pack(side="left")

        kanan = tk.Frame(hdr, bg=PANEL)
        kanan.pack(side="right", padx=16)
        buat_tombol(kanan, "📁  Pilih Gambar",        self._pilih_gambar,
                    warna="#21262d", lebar=16).pack(side="left", padx=4)
        buat_tombol(kanan, "📄  Upload Ground Truth",  self._pilih_gt,
                    warna="#21262d", lebar=22).pack(side="left", padx=4)
        buat_tombol(kanan, "▶  Jalankan",              self._jalankan,
                    warna=AKSEN,     lebar=14).pack(side="left", padx=4)
        buat_tombol(kanan, "↺  Reset",                 self._reset,
                    warna="#21262d", lebar=10).pack(side="left", padx=4)

    def _bangun_panel_gambar_asli(self, parent):
        f = tk.Frame(parent, bg=PANEL)
        f.grid(row=0, column=0, sticky="nsew", padx=(0, 4), pady=4)
        f.grid_propagate(False)

        self.canvas_asli = ZoomableCanvas(f)

        hdr = tk.Frame(f, bg=PANEL)
        hdr.pack(fill="x", padx=12, pady=(10, 6))
        tk.Frame(hdr, bg=AKSEN, width=3, height=16).pack(side="left", padx=(0, 8))
        tk.Label(hdr, text="Gambar Asli", font=F_JUDUL, bg=PANEL, fg=PUTIH).pack(side="left")
        tk.Button(hdr, text="⊡ Reset Zoom", command=self.canvas_asli.reset_zoom,
                  font=F_KECIL, bg="#21262d", fg=TEKS2, activebackground="#30363d",
                  activeforeground=TEKS, relief="flat", bd=0, padx=6, pady=2,
                  cursor="hand2").pack(side="right")

        self._cvs_nama_asli = tk.Canvas(hdr, bg=PANEL, height=18, highlightthickness=0, cursor="fleur")
        self._cvs_nama_asli.pack(side="left", fill="x", expand=True, padx=(10, 4))
        self._inner_nama_asli = tk.Frame(self._cvs_nama_asli, bg=PANEL)
        self._cvs_nama_asli.create_window((0, 0), window=self._inner_nama_asli, anchor="nw")
        self.lbl_info_asli = tk.Label(self._inner_nama_asli, text="",
                                      font=F_KECIL, bg=PANEL, fg=TEKS2, cursor="fleur")
        self.lbl_info_asli.pack(side="left")
        self._inner_nama_asli.bind("<Configure>", lambda e:
            self._cvs_nama_asli.configure(scrollregion=self._cvs_nama_asli.bbox("all")))
        self._cvs_nama_asli._dx = 0
        self._cvs_nama_asli.bind("<ButtonPress-1>",
            lambda e: setattr(self._cvs_nama_asli, "_dx", e.x))
        self._cvs_nama_asli.bind("<B1-Motion>", self._drag_nama_asli)
        self.lbl_info_asli.bind("<ButtonPress-1>",
            lambda e: setattr(self._cvs_nama_asli, "_dx", e.x))
        self.lbl_info_asli.bind("<B1-Motion>", self._drag_nama_asli)

        self.canvas_asli.configure(width=10, height=10)
        self.canvas_asli.pack(fill="both", expand=True, padx=12, pady=(0, 8))

    def _bangun_panel_gambar_prep(self, parent):
        f = tk.Frame(parent, bg=PANEL)
        f.grid(row=0, column=1, sticky="nsew", padx=(4, 0), pady=4)
        f.grid_propagate(False)

        self.canvas_prep = ZoomableCanvas(f)
        judul_seksi(f, "Hasil Preprocessing Adaptif", warna_aksen=AKSEN2,
                    canvas_zoom=self.canvas_prep)
        self.canvas_prep.configure(width=10, height=10)
        self.canvas_prep.pack(fill="both", expand=True, padx=12, pady=(0, 8))
        self._badge = {}

    def _bangun_panel_kontrol_analisis(self, parent):
        f = tk.Frame(parent, bg=PANEL)
        f.grid(row=0, column=2, sticky="nsew", padx=(4, 0), pady=4)
        f.grid_propagate(False)

        judul_seksi(f, "Kontrol, Analisis & Evaluasi")

        baris_ctrl = tk.Frame(f, bg=PANEL)
        baris_ctrl.pack(fill="x", padx=12, pady=(0, 6))
        tk.Label(baris_ctrl, text="Klik untuk aktif/nonaktif:",
                 font=F_KECIL, bg=PANEL, fg=TEKS2).pack(side="left")

        self.frame_badge = tk.Frame(f, bg=PANEL)
        self.frame_badge.pack(padx=12, fill="x")

        self._badge = {}
        baris1 = tk.Frame(self.frame_badge, bg=PANEL)
        baris1.pack(fill="x", pady=(0, 4))
        baris2 = tk.Frame(self.frame_badge, bg=PANEL)
        baris2.pack(fill="x")

        for t in ["Grayscale", "CLAHE", "Adaptive Gaussian"]:
            btn = tk.Button(baris1, text=t, font=F_BADGE, bg=TEKS3, fg="#555",
                            activebackground=TEKS3, activeforeground="#555",
                            relief="flat", bd=0, padx=10, pady=5,
                            cursor="arrow", highlightthickness=0,
                            command=lambda nama=t: self._toggle_teknik(nama))
            btn.pack(side="left", padx=(0, 4))
            self._badge[t] = btn

        btn_lr = tk.Button(baris2, text="Line Removal", font=F_BADGE,
                           bg=TEKS3, fg="#555", activebackground=TEKS3,
                           activeforeground="#555", relief="flat", bd=0,
                           padx=10, pady=5, cursor="arrow", highlightthickness=0,
                           command=lambda: self._toggle_teknik("Line Removal"))
        btn_lr.pack(side="left", padx=(0, 4))
        self._badge["Line Removal"] = btn_lr

        self._btn_adaptif = tk.Button(
            baris2, text="Adaptif Preprocessing", font=F_BADGE,
            bg=TEKS3, fg="#555", activebackground=TEKS3, activeforeground="#555",
            relief="flat", bd=0, padx=10, pady=5, cursor="arrow", highlightthickness=0,
            command=self._toggle_adaptif)
        self._btn_adaptif.pack(side="left", padx=(0, 4))

        tk.Frame(f, bg=BORDER, height=1).pack(fill="x", padx=12, pady=10)

        tk.Label(f, text="Analisis Gambar", font=F_JUDUL, bg=PANEL, fg=PUTIH
                 ).pack(padx=12, anchor="w", pady=(0, 6))
        self._bangun_tabel_analisis(f, [
            ("intensitas",  "Intensitas\nCahaya"),
            ("rasio_garis", "Deteksi\nGaris"),
            ("ada_garis",   "Garis\nTabel"),
        ])

        tk.Frame(f, bg=BORDER, height=1).pack(fill="x", padx=12, pady=10)

        tk.Label(f, text="Evaluasi OCR", font=F_JUDUL, bg=PANEL, fg=PUTIH
                 ).pack(padx=12, anchor="w", pady=(0, 6))
        self._bangun_tabel_metrik(f, [("cer", "CER (%)"), ("wer", "WER (%)")])

    def _bangun_tabel_analisis(self, parent, rows):
        hdr = tk.Frame(parent, bg=PANEL)
        hdr.pack(fill="x", padx=12)
        hdr.columnconfigure(0, weight=3)
        hdr.columnconfigure(1, weight=2, minsize=64)
        hdr.columnconfigure(2, weight=2, minsize=64)
        tk.Label(hdr, text="Metrik",  font=F_KECIL, bg=PANEL, fg=TEKS2).grid(row=0, column=0, sticky="w")
        tk.Label(hdr, text="Sebelum", font=F_KECIL, bg=PANEL, fg=TEKS2).grid(row=0, column=1, sticky="w", padx=(4,2))
        tk.Label(hdr, text="Sesudah", font=F_KECIL, bg=PANEL, fg=TEKS2).grid(row=0, column=2, sticky="w", padx=(2,0))
        tk.Frame(parent, bg=BORDER, height=1).pack(fill="x", padx=12, pady=3)

        self._analisis_labels = {}
        for key, nama in rows:
            rf = tk.Frame(parent, bg=PANEL)
            rf.pack(fill="x", padx=12, pady=3)
            rf.columnconfigure(0, weight=3)
            rf.columnconfigure(1, weight=2, minsize=64)
            rf.columnconfigure(2, weight=2, minsize=64)
            tk.Label(rf, text=nama, font=F_KECIL, bg=PANEL,
                     fg=TEKS, justify="left").grid(row=0, column=0, sticky="w")
            lb = tk.Label(rf, text="—", font=F_KECIL, bg=CARD, fg=TEKS2, anchor="center", pady=5)
            lb.grid(row=0, column=1, sticky="ew", padx=(4, 2))
            la = tk.Label(rf, text="—", font=F_KECIL, bg=CARD, fg=TEKS2, anchor="center", pady=5)
            la.grid(row=0, column=2, sticky="ew", padx=(2, 0))
            self._analisis_labels[key] = (lb, la)

    def _bangun_tabel_metrik(self, parent, rows):
        hdr = tk.Frame(parent, bg=PANEL)
        hdr.pack(fill="x", padx=12)
        hdr.columnconfigure(0, weight=3)
        hdr.columnconfigure(1, weight=2, minsize=64)
        hdr.columnconfigure(2, weight=2, minsize=64)
        tk.Label(hdr, text="Metrik",  font=F_KECIL, bg=PANEL, fg=TEKS2).grid(row=0, column=0, sticky="w")
        tk.Label(hdr, text="Sebelum", font=F_KECIL, bg=PANEL, fg=TEKS2).grid(row=0, column=1, sticky="w", padx=(4,2))
        tk.Label(hdr, text="Sesudah", font=F_KECIL, bg=PANEL, fg=TEKS2).grid(row=0, column=2, sticky="w", padx=(2,0))
        tk.Frame(parent, bg=BORDER, height=1).pack(fill="x", padx=12, pady=3)

        self._metrik_labels = {}
        for key, nama in rows:
            rf = tk.Frame(parent, bg=PANEL)
            rf.pack(fill="x", padx=12, pady=3)
            rf.columnconfigure(0, weight=3)
            rf.columnconfigure(1, weight=2, minsize=64)
            rf.columnconfigure(2, weight=2, minsize=64)
            tk.Label(rf, text=nama, font=F_KECIL, bg=PANEL,
                     fg=TEKS, justify="left").grid(row=0, column=0, sticky="w")
            lb = tk.Label(rf, text="—", font=F_KECIL, bg=CARD, fg=TEKS2, anchor="center", pady=5)
            lb.grid(row=0, column=1, sticky="ew", padx=(4, 2))
            la = tk.Label(rf, text="—", font=F_KECIL, bg=CARD, fg=TEKS2, anchor="center", pady=5)
            la.grid(row=0, column=2, sticky="ew", padx=(2, 0))
            li = tk.Label(rf, text="", font=F_KECIL, bg=PANEL, fg=TEKS2, anchor="w")
            li.grid(row=1, column=0, columnspan=3, sticky="w", pady=(1, 0))
            self._metrik_labels[key] = (lb, la, li)

    def _bangun_panel_gt(self, parent):
        f = tk.Frame(parent, bg=PANEL)
        f.grid(row=0, column=0, sticky="nsew", padx=(0, 4), pady=4)

        hdr = tk.Frame(f, bg=PANEL)
        hdr.pack(fill="x", padx=12, pady=(10, 6))
        tk.Frame(hdr, bg=AKSEN, width=3, height=16).pack(side="left", padx=(0, 8))
        tk.Label(hdr, text="Ground Truth", font=F_JUDUL, bg=PANEL, fg=PUTIH).pack(side="left")

        self._cvs_nama_gt = tk.Canvas(hdr, bg=PANEL, height=18, highlightthickness=0, cursor="fleur")
        self._cvs_nama_gt.pack(side="left", fill="x", expand=True, padx=(10, 0))
        self._inner_nama_gt = tk.Frame(self._cvs_nama_gt, bg=PANEL)
        self._cvs_nama_gt.create_window((0, 0), window=self._inner_nama_gt, anchor="nw")
        self.lbl_gt_file = tk.Label(self._inner_nama_gt, text="",
                                    font=F_KECIL, bg=PANEL, fg=TEKS2, cursor="fleur")
        self.lbl_gt_file.pack(side="left")
        self._inner_nama_gt.bind("<Configure>", lambda e:
            self._cvs_nama_gt.configure(scrollregion=self._cvs_nama_gt.bbox("all")))
        self._cvs_nama_gt._dx = 0
        self._cvs_nama_gt.bind("<ButtonPress-1>",
            lambda e: setattr(self._cvs_nama_gt, "_dx", e.x))
        self._cvs_nama_gt.bind("<B1-Motion>", self._drag_nama_gt)
        self.lbl_gt_file.bind("<ButtonPress-1>",
            lambda e: setattr(self._cvs_nama_gt, "_dx", e.x))
        self.lbl_gt_file.bind("<B1-Motion>", self._drag_nama_gt)

        inner = tk.Frame(f, bg=CARD)
        inner.pack(fill="both", expand=True, padx=12, pady=(0, 8))
        self.txt_gt = self._buat_textbox(inner, bg=CARD)
        self.txt_gt.pack(side="left", fill="both", expand=True)
        sb = ttk.Scrollbar(inner, orient="vertical", command=self.txt_gt.yview)
        sb.pack(side="right", fill="y")
        self.txt_gt.configure(yscrollcommand=sb.set)

    def _bangun_panel_ocr_asli(self, parent):
        f = tk.Frame(parent, bg=PANEL)
        f.grid(row=0, column=1, sticky="nsew", padx=4, pady=4)
        judul_seksi(f, "OCR Tanpa Preprocessing")
        inner = tk.Frame(f, bg=CARD)
        inner.pack(fill="both", expand=True, padx=12, pady=(0, 8))
        self.txt_ocr_asli = self._buat_textbox(inner, bg=CARD)
        self.txt_ocr_asli.pack(side="left", fill="both", expand=True)
        sb = ttk.Scrollbar(inner, orient="vertical", command=self.txt_ocr_asli.yview)
        sb.pack(side="right", fill="y")
        self.txt_ocr_asli.configure(yscrollcommand=sb.set)

    def _bangun_panel_ocr_prep(self, parent):
        f = tk.Frame(parent, bg=PANEL)
        f.grid(row=0, column=2, sticky="nsew", padx=(4, 0), pady=4)
        judul_seksi(f, "OCR Setelah Preprocessing", warna_aksen=AKSEN2)
        inner = tk.Frame(f, bg=CARD)
        inner.pack(fill="both", expand=True, padx=12, pady=(0, 8))
        self.txt_ocr_prep = self._buat_textbox(inner, bg=CARD)
        self.txt_ocr_prep.pack(side="left", fill="both", expand=True)
        sb = ttk.Scrollbar(inner, orient="vertical", command=self.txt_ocr_prep.yview)
        sb.pack(side="right", fill="y")
        self.txt_ocr_prep.configure(yscrollcommand=sb.set)

    def _buat_textbox(self, parent, bg=CARD):
        return tk.Text(parent, font=F_MONO, bg=bg, fg=TEKS,
                       insertbackground=TEKS, relief="flat", bd=0,
                       wrap="word", padx=10, pady=10, state="disabled")

    def _bangun_scroll_container(self):
        wrap = tk.Frame(self, bg=BG)
        wrap.pack(fill="both", expand=True)
        self._scrollbar = ttk.Scrollbar(wrap, orient="vertical")
        self._scrollbar.pack(side="right", fill="y")
        self._scroll_canvas = tk.Canvas(wrap, bg=BG, highlightthickness=0,
                                        yscrollcommand=self._scrollbar.set)
        self._scroll_canvas.pack(side="left", fill="both", expand=True)
        self._scrollbar.configure(command=self._scroll_canvas.yview)
        self._scroll_inner  = tk.Frame(self._scroll_canvas, bg=BG)
        self._scroll_win_id = self._scroll_canvas.create_window(
            (0, 0), window=self._scroll_inner, anchor="nw")
        self._scroll_inner.bind("<Configure>",  self._on_scroll_inner_configure)
        self._scroll_canvas.bind("<Configure>", self._on_scroll_canvas_configure)
        self._scroll_canvas.bind("<MouseWheel>", self._on_mousewheel)
        self._scroll_canvas.bind("<Button-4>",   self._on_scroll_up)
        self._scroll_canvas.bind("<Button-5>",   self._on_scroll_down)
        self._scroll_inner.bind("<MouseWheel>",  self._on_mousewheel)
        self.bind_all("<MouseWheel>", self._on_mousewheel)

    def _on_scroll_inner_configure(self, event):
        self._scroll_canvas.configure(scrollregion=self._scroll_canvas.bbox("all"))

    def _on_scroll_canvas_configure(self, event):
        w = event.width
        self._scroll_canvas.itemconfigure(self._scroll_win_id, width=w)
        lebar = max(w - 24, 200)
        try:
            self.frame_atas.configure(width=lebar)
            self.frame_bawah.configure(width=lebar)
        except Exception:
            pass

    def _on_mousewheel(self, event):
        if not self.winfo_ismapped():
            return
        if isinstance(event.widget, ZoomableCanvas):
            return
        self._scroll_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _on_scroll_up(self,   event):
        if self.winfo_ismapped():
            self._scroll_canvas.yview_scroll(-1, "units")
    def _on_scroll_down(self, event):
        if self.winfo_ismapped():
            self._scroll_canvas.yview_scroll(1, "units")

    def _drag_nama_asli(self, event):
        dx = self._cvs_nama_asli._dx - event.x
        self._cvs_nama_asli.xview_scroll(int(dx / 2), "units")
        self._cvs_nama_asli._dx = event.x

    def _drag_nama_gt(self, event):
        dx = self._cvs_nama_gt._dx - event.x
        self._cvs_nama_gt.xview_scroll(int(dx / 2), "units")
        self._cvs_nama_gt._dx = event.x

    def _pilih_gambar(self):
        path = filedialog.askopenfilename(filetypes=[("Gambar", "*.png")])
        if not path:
            return
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            messagebox.showerror("Error", "Gagal membaca gambar.")
            return
        self.img_asli = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        self.canvas_asli.set_image(self.img_asli)
        self.lbl_info_asli.configure(text=f"—  {os.path.basename(path)}")

    def _pilih_gt(self):
        path = filedialog.askopenfilename(
            filetypes=[("Teks", "*.txt"), ("Semua file", "*.*")])
        if not path:
            return
        try:
            with open(path, encoding="utf-8") as f:
                self.teks_gt = f.read()
        except Exception:
            with open(path, encoding="latin-1") as f:
                self.teks_gt = f.read()
        self.lbl_gt_file.configure(
            text=f"—  {os.path.basename(path)}  ({len(self.teks_gt)} karakter)")
        self._isi_teks(self.txt_gt, self.teks_gt)

    def _jalankan(self):
        if self.img_asli is None:
            messagebox.showinfo("Info", "Pilih gambar terlebih dahulu.")
            return
        self.winfo_toplevel().configure(cursor="watch")
        self.update()

        self.teks_ocr_asli = run_ocr(self.img_asli)
        self._isi_teks(self.txt_ocr_asli, self.teks_ocr_asli)

        self.img_prep, self.log_teknik = pipeline_adaptive(self.img_asli)
        self.canvas_prep.set_image(self.img_prep, gray=True)
        self.teks_ocr_prep = run_ocr(self.img_prep)
        self._isi_teks(self.txt_ocr_prep, self.teks_ocr_prep)

        a_before = analisis_gambar(self.img_asli)
        a_after  = analisis_gambar(self.img_prep)
        self._analisis_awal = a_before
        self._update_analisis(a_before, a_after)

        self.aktif_teknik = {
            "Grayscale":         True,
            "CLAHE":             bool(self.log_teknik.get("clahe")),
            "Adaptive Gaussian": True,
            "Line Removal":      bool(self.log_teknik.get("line_removal")),
        }
        self.aktif_awal   = dict(self.aktif_teknik)
        self.mode_adaptif = True
        self.dimatikan.clear()
        self._perbarui_badge()
        self._perbarui_btn_adaptif()

        if self.teks_gt.strip():
            m_asli = hitung_cer_wer(self.teks_ocr_asli, self.teks_gt)
            m_prep = hitung_cer_wer(self.teks_ocr_prep, self.teks_gt)
            self._update_metrik(m_asli, m_prep)
        else:
            self._reset_metrik_labels("Upload GT untuk evaluasi")

        self.winfo_toplevel().configure(cursor="")

    def _toggle_adaptif(self):
        if not self.aktif_awal:
            return
        if self.mode_adaptif:
            self.mode_adaptif = False
        else:
            self.aktif_teknik = dict(self.aktif_awal)
            self.mode_adaptif = True
            self._perbarui_badge()
            self.update()
            self._rerun_prep()
        self._perbarui_btn_adaptif()

    def _toggle_teknik(self, nama: str):
        if not self.aktif_teknik:
            return
        sekarang = self.aktif_teknik.get(nama, False)
        if sekarang:
            self.aktif_teknik[nama] = False
            if nama == "Grayscale":
                for k in ["CLAHE", "Adaptive Gaussian", "Line Removal"]:
                    self.aktif_teknik[k] = False
            elif nama == "Adaptive Gaussian":
                self.aktif_teknik["Line Removal"] = False
        else:
            if nama == "Line Removal" and not self.aktif_teknik.get("Adaptive Gaussian", False):
                return
            if nama != "Grayscale" and not self.aktif_teknik.get("Grayscale", False):
                return
            self.aktif_teknik[nama] = True
        self.mode_adaptif = False
        self._perbarui_btn_adaptif()
        self._perbarui_badge()
        self.update()
        self._rerun_prep()

    def _rerun_prep(self):
        if self.img_asli is None or not self.log_teknik:
            return
        self.winfo_toplevel().configure(cursor="watch")
        self.update()

        img_tampil, img_ocr = pipeline_manual(self.img_asli, self.aktif_teknik)
        self.img_prep = img_tampil
        gray_mode = self.aktif_teknik.get("Grayscale", True)
        self.canvas_prep.set_image(img_tampil, gray=gray_mode)

        self.teks_ocr_prep = run_ocr(img_ocr)
        self._isi_teks(self.txt_ocr_prep, self.teks_ocr_prep)

        if self.teks_gt.strip():
            m_asli = hitung_cer_wer(self.teks_ocr_asli, self.teks_gt)
            m_prep = hitung_cer_wer(self.teks_ocr_prep, self.teks_gt)
            self._update_metrik(m_asli, m_prep)

        a_after = analisis_gambar(img_ocr)
        if self._analisis_awal is not None:
            a_before = self._analisis_awal
        else:
            a_before = analisis_gambar(self.img_asli)
        self._update_analisis(a_before, a_after)
        self.winfo_toplevel().configure(cursor="")

    def _perbarui_btn_adaptif(self):
        if not self.aktif_awal:
            self._btn_adaptif.configure(bg=TEKS3, fg="#555",
                                        activebackground=TEKS3, activeforeground="#555",
                                        cursor="arrow")
            return
        if self.mode_adaptif:
            self._btn_adaptif.configure(bg=AKSEN2, fg=PUTIH,
                                        activebackground=AKSEN2, activeforeground=PUTIH,
                                        cursor="hand2")
        else:
            self._btn_adaptif.configure(bg=TEKS3, fg="#888",
                                        activebackground=TEKS3, activeforeground="#888",
                                        cursor="hand2")

    def _perbarui_badge(self):
        for nama, btn in self._badge.items():
            if not self.aktif_teknik:
                btn.configure(bg=TEKS3, fg="#555",
                               activebackground=TEKS3, activeforeground="#555", cursor="arrow")
                continue
            aktif      = self.aktif_teknik.get(nama, False)
            gray_aktif = self.aktif_teknik.get("Grayscale", False)
            bin_aktif  = self.aktif_teknik.get("Adaptive Gaussian", False)
            if nama == "Grayscale":
                bisa_klik = True
            elif nama == "Line Removal":
                bisa_klik = aktif or (gray_aktif and bin_aktif)
            else:
                bisa_klik = aktif or gray_aktif
            if aktif:
                btn.configure(bg=AKSEN, fg=PUTIH, activebackground=AKSEN, activeforeground=PUTIH,
                               cursor="hand2" if bisa_klik else "arrow")
            else:
                btn.configure(bg=TEKS3, fg="#888", activebackground=TEKS3, activeforeground="#888",
                               cursor="hand2" if bisa_klik else "arrow")

    def _update_metrik(self, m_asli: dict, m_prep: dict):
        def fmt(v): return f"{v:.1f}" if v is not None else "—"
        def ind(a, p):
            if a is None or p is None: return "Upload GT untuk evaluasi", TEKS2
            d = p - a
            if abs(d) < 0.01: return "= tidak berubah", TEKS2
            elif d < 0: return f"▼ {abs(d):.2f}% membaik", HIJAU
            else:       return f"▲ {d:.2f}% memburuk", MERAH

        for key, (m_a, m_p) in [("cer", (m_asli["cer"], m_prep["cer"])),
                                  ("wer", (m_asli["wer"], m_prep["wer"]))]:
            lb, la, li = self._metrik_labels[key]
            lb.configure(text=fmt(m_a), fg=KUNING if m_a is not None else TEKS2)
            if m_a is None or m_p is None:
                warna_p = TEKS2
            elif m_p < m_a:
                warna_p = HIJAU
            elif m_p > m_a:
                warna_p = MERAH
            else:
                warna_p = TEKS2
            la.configure(text=fmt(m_p), fg=warna_p)
            teks, warna = ind(m_a, m_p)
            li.configure(text=teks, fg=warna)

    def _reset_metrik_labels(self, pesan=""):
        for key in ["cer", "wer"]:
            lb, la, li = self._metrik_labels[key]
            lb.configure(text="—", fg=TEKS2)
            la.configure(text="—", fg=TEKS2)
            li.configure(text=pesan, fg=TEKS2)

    def _update_analisis(self, before: dict, after: dict):
        def wi(v): return HIJAU if 100 <= v <= 200 else (KUNING if 80 <= v <= 220 else MERAH)
        def wr(v): return MERAH if v > 0.3 else HIJAU

        for key, (lb, la) in self._analisis_labels.items():
            if key == "ada_garis":
                for lbl, src in [(lb, before), (la, after)]:
                    ada = src.get("ada_garis", False)
                    lbl.configure(text="Ada" if ada else "Tidak Ada",
                                  fg=MERAH if ada else HIJAU)
            elif key == "intensitas":
                vb, va = before.get(key, 0), after.get(key, 0)
                lb.configure(text=f"{vb:.2f}", fg=wi(vb))
                la.configure(text=f"{va:.2f}", fg=HIJAU if va > vb else MERAH)
            else:
                vb, va = before.get(key, 0), after.get(key, 0)
                lb.configure(text=f"{vb:.2f}", fg=wr(vb))
                la.configure(text=f"{va:.2f}", fg=wr(va))

    def _isi_teks(self, widget, teks):
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", teks)
        widget.configure(state="disabled")

    def _reset(self):
        self.img_asli = self.img_prep = None
        self.teks_gt = self.teks_ocr_asli = self.teks_ocr_prep = ""
        self.log_teknik   = {}
        self.mode_adaptif = False
        self._analisis_awal = None
        self.canvas_asli.clear()
        self.canvas_prep.clear()
        self.lbl_info_asli.configure(text="")
        self.lbl_gt_file.configure(text="")
        for w in [self.txt_gt, self.txt_ocr_asli, self.txt_ocr_prep]:
            self._isi_teks(w, "")
        self._reset_metrik_labels()
        for key, (lb, la) in self._analisis_labels.items():
            lb.configure(text="—", fg=TEKS2)
            la.configure(text="—", fg=TEKS2)
        self.dimatikan.clear()
        self.aktif_teknik.clear()
        self.aktif_awal.clear()
        for nama, btn in self._badge.items():
            btn.configure(bg=TEKS3, fg="#555",
                          activebackground=TEKS3, activeforeground="#555", cursor="arrow")
        self._perbarui_btn_adaptif()


class EvaluasiFrame(tk.Frame):

    def __init__(self, parent, **kw):
        super().__init__(parent, bg=BG, **kw)
        self.gambar_list = []
        self.gt_dict     = {}
        self.gt_files    = []
        self._queue      = queue.Queue()
        self._running    = False
        self._csv_data   = None
        self._bangun_ui()

    def _bangun_ui(self):
        self._bangun_topbar()

        content = tk.Frame(self, bg=BG)
        content.pack(fill="both", expand=True)

        self.frame_atas = tk.Frame(content, bg=BG, height=265)
        self.frame_atas.pack(fill="x", padx=12, pady=(0, 4))
        self.frame_atas.pack_propagate(False)
        self.frame_atas.columnconfigure(0, weight=1, uniform="atas", minsize=180)
        self.frame_atas.columnconfigure(1, weight=1, uniform="atas", minsize=180)
        self.frame_atas.columnconfigure(2, weight=1, uniform="atas", minsize=180)
        self.frame_atas.rowconfigure(0, weight=1, minsize=265)

        self._bangun_card_gambar(self.frame_atas)
        self._bangun_card_gt(self.frame_atas)
        self._bangun_card_status(self.frame_atas)

        self.frame_bawah = tk.Frame(content, bg=BG)
        self.frame_bawah.pack(fill="both", expand=True, padx=12, pady=(0, 8))

        self._bangun_card_terminal_eval(self.frame_bawah)

    def _bangun_card_gambar(self, parent):
        f = tk.Frame(parent, bg=PANEL)
        f.grid(row=0, column=0, sticky="nsew", padx=(0, 4), pady=4)

        hdr = tk.Frame(f, bg=PANEL)
        hdr.pack(fill="x", padx=12, pady=(10, 6))
        tk.Frame(hdr, bg=AKSEN, width=3, height=16).pack(side="left", padx=(0, 8))
        tk.Label(hdr, text="Gambar Input", font=F_JUDUL, bg=PANEL, fg=PUTIH).pack(side="left")
        self._lbl_g_count = tk.Label(hdr, text="(0)", font=F_KECIL, bg=PANEL, fg=TEKS2)
        self._lbl_g_count.pack(side="left", padx=4)

        frame_lb = tk.Frame(f, bg=CARD)
        frame_lb.pack(fill="both", expand=True, padx=12, pady=(0, 4))
        sb_g = ttk.Scrollbar(frame_lb, orient="vertical")
        sb_g.pack(side="right", fill="y", pady=4)
        self.lb_gambar = tk.Listbox(
            frame_lb, font=F_KECIL, bg=CARD, fg=TEKS,
            selectbackground=AKSEN, selectforeground=PUTIH,
            relief="flat", bd=0, activestyle="none",
            selectmode="extended", yscrollcommand=sb_g.set)
        self.lb_gambar.pack(side="left", fill="both", expand=True, padx=(6, 0), pady=4)
        sb_g.configure(command=self.lb_gambar.yview)

        btn_f = tk.Frame(f, bg=PANEL)
        btn_f.pack(fill="x", padx=12, pady=(0, 10))
        buat_tombol(btn_f, "+ Tambah", self._pilih_gambar,
                    warna="#21262d", lebar=10).pack(side="left", padx=(0, 4))
        buat_tombol(btn_f, "− Hapus", self._hapus_gambar,
                    warna="#21262d", lebar=10).pack(side="left")

    def _bangun_card_gt(self, parent):
        f = tk.Frame(parent, bg=PANEL)
        f.grid(row=0, column=1, sticky="nsew", padx=4, pady=4)

        hdr = tk.Frame(f, bg=PANEL)
        hdr.pack(fill="x", padx=12, pady=(10, 6))
        tk.Frame(hdr, bg=AKSEN2, width=3, height=16).pack(side="left", padx=(0, 8))
        tk.Label(hdr, text="Ground Truth", font=F_JUDUL, bg=PANEL, fg=PUTIH).pack(side="left")
        self._lbl_gt_count = tk.Label(hdr, text="(0)", font=F_KECIL, bg=PANEL, fg=TEKS2)
        self._lbl_gt_count.pack(side="left", padx=4)

        frame_lb = tk.Frame(f, bg=CARD)
        frame_lb.pack(fill="both", expand=True, padx=12, pady=(0, 4))
        sb_gt = ttk.Scrollbar(frame_lb, orient="vertical")
        sb_gt.pack(side="right", fill="y", pady=4)
        self.lb_gt = tk.Listbox(
            frame_lb, font=F_KECIL, bg=CARD, fg=TEKS,
            selectbackground=AKSEN2, selectforeground=PUTIH,
            relief="flat", bd=0, activestyle="none",
            selectmode="extended", yscrollcommand=sb_gt.set)
        self.lb_gt.pack(side="left", fill="both", expand=True, padx=(6, 0), pady=4)
        sb_gt.configure(command=self.lb_gt.yview)

        btn_f = tk.Frame(f, bg=PANEL)
        btn_f.pack(fill="x", padx=12, pady=(0, 10))
        buat_tombol(btn_f, "+ Tambah", self._pilih_gt,
                    warna="#21262d", lebar=10).pack(side="left", padx=(0, 4))
        buat_tombol(btn_f, "− Hapus", self._hapus_gt,
                    warna="#21262d", lebar=10).pack(side="left")

    def _bangun_card_status(self, parent):
        f = tk.Frame(parent, bg=PANEL)
        f.grid(row=0, column=2, sticky="nsew", padx=(4, 0), pady=4)

        hdr = tk.Frame(f, bg=PANEL)
        hdr.pack(fill="x", padx=12, pady=(10, 6))
        tk.Frame(hdr, bg=KUNING, width=3, height=16).pack(side="left", padx=(0, 8))
        tk.Label(hdr, text="Status & Teknik", font=F_JUDUL, bg=PANEL, fg=PUTIH).pack(side="left")

        tk.Frame(f, bg=BORDER, height=1).pack(fill="x", padx=12, pady=(0, 8))

        tk.Label(f, text="Status Pencocokan", font=F_JUDUL, bg=PANEL, fg=PUTIH
                 ).pack(padx=12, anchor="w", pady=(0, 4))
        self._lbl_status = tk.Label(f, text="Belum ada file dipilih.",
                                    font=F_KECIL, bg=PANEL, fg=TEKS2,
                                    wraplength=210, justify="left", anchor="nw")
        self._lbl_status.pack(padx=12, anchor="w")

        tk.Frame(f, bg=BORDER, height=1).pack(fill="x", padx=12, pady=(10, 8))

        tk.Label(f, text="Teknik yang Diuji", font=F_JUDUL, bg=PANEL, fg=PUTIH
                 ).pack(padx=12, anchor="w", pady=(0, 4))
        for t in EVAL_TEKNIK:
            tk.Label(f, text=f"  • {t}", font=F_KECIL, bg=PANEL, fg=TEKS2, anchor="w"
                     ).pack(padx=12, fill="x")

    def _bangun_card_terminal_eval(self, parent):
        f = tk.Frame(parent, bg=PANEL)
        f.pack(fill="both", expand=True, pady=4)

        hdr = tk.Frame(f, bg=PANEL)
        hdr.pack(fill="x", padx=12, pady=(10, 6))
        tk.Frame(hdr, bg=AKSEN2, width=3, height=16).pack(side="left", padx=(0, 8))
        tk.Label(hdr, text="Output Terminal", font=F_JUDUL, bg=PANEL, fg=PUTIH).pack(side="left")

        tk.Frame(f, bg=BORDER, height=1).pack(fill="x")

        term_f = tk.Frame(f, bg=TERM_BG)
        term_f.pack(fill="both", expand=True)

        self.terminal = tk.Text(
            term_f, font=F_TERM, bg=TERM_BG, fg=TEKS,
            insertbackground=TEKS, relief="flat", bd=0,
            wrap="none", padx=14, pady=12, state="disabled")
        sb_y = ttk.Scrollbar(term_f, orient="vertical",   command=self.terminal.yview)
        sb_x = ttk.Scrollbar(term_f, orient="horizontal", command=self.terminal.xview)
        self.terminal.configure(yscrollcommand=sb_y.set, xscrollcommand=sb_x.set)
        sb_y.pack(side="right",  fill="y")
        sb_x.pack(side="bottom", fill="x")
        self.terminal.pack(fill="both", expand=True)

        self.terminal.tag_configure("normal",  foreground=TEKS)
        self.terminal.tag_configure("header",  foreground=AKSEN2, font=("Consolas", 9, "bold"))
        self.terminal.tag_configure("aksen",   foreground=AKSEN,  font=("Consolas", 9, "bold"))
        self.terminal.tag_configure("aksen2",  foreground=AKSEN2)
        self.terminal.tag_configure("hijau",   foreground=HIJAU)
        self.terminal.tag_configure("merah",   foreground=MERAH)
        self.terminal.tag_configure("kuning",  foreground=KUNING)
        self.terminal.tag_configure("abu",     foreground=TEKS3)
        self.terminal.tag_configure("sukses",  foreground=HIJAU, font=("Consolas", 9, "bold"))
        self.terminal.tag_configure("bold",    font=("Consolas", 9, "bold"))

    def _bangun_topbar(self):
        bar = tk.Frame(self, bg=PANEL, height=58)
        bar.pack(fill="x")
        bar.pack_propagate(False)

        kiri = tk.Frame(bar, bg=PANEL)
        kiri.pack(side="left", padx=20, pady=10)
        tk.Frame(kiri, bg=AKSEN2, width=4, height=30).pack(side="left", padx=(0, 10))
        tk.Label(kiri, text="Pipeline Evaluasi Preprocessing OCR",
                 font=("Segoe UI", 13, "bold"), bg=PANEL, fg=PUTIH).pack(side="left")

        kanan = tk.Frame(bar, bg=PANEL)
        kanan.pack(side="right", padx=16)

        self._btn_jalankan = buat_tombol(kanan, "▶  Jalankan Evaluasi",
                                          self._jalankan, warna=AKSEN, lebar=20)
        self._btn_jalankan.pack(side="left", padx=4)

        self._btn_csv = buat_tombol(kanan, "⬇  Simpan CSV",
                                     self._simpan_csv, warna=TEKS3, lebar=16)
        self._btn_csv.pack(side="left", padx=4)
        self._btn_csv.configure(state="disabled")

        buat_tombol(kanan, "🗑  Bersihkan",
                    self._clear_terminal_ui, warna="#21262d", lebar=14).pack(side="left", padx=4)

        buat_tombol(kanan, "↺  Reset",
                    self._reset, warna="#21262d", lebar=10).pack(side="left", padx=4)

        self._lbl_run_status = tk.Label(bar, text="", font=F_KECIL, bg=PANEL, fg=TEKS2)
        self._lbl_run_status.pack(side="right", padx=12)

    def _append_terminal(self, text: str, tag: str = "normal"):
        self.terminal.configure(state="normal")
        self.terminal.insert("end", text, tag)
        self.terminal.see("end")
        self.terminal.configure(state="disabled")

    def _clear_terminal_ui(self):
        self.terminal.configure(state="normal")
        self.terminal.delete("1.0", "end")
        self.terminal.configure(state="disabled")

    def _pilih_gambar(self):
        paths = filedialog.askopenfilenames(
            title="Pilih Gambar Input",
            filetypes=[("PNG", "*.png"), ("Gambar", "*.jpg *.jpeg *.png *.bmp"),
                       ("Semua file", "*.*")])
        added = 0
        for path in paths:
            fname = os.path.basename(path)
            if not any(p == path for p, _ in self.gambar_list):
                self.gambar_list.append((path, fname))
                self.lb_gambar.insert("end", fname)
                added += 1
        if added:
            self._update_status()

    def _pilih_gt(self):
        paths = filedialog.askopenfilenames(
            title="Pilih File Ground Truth",
            filetypes=[("Teks", "*.txt"), ("Semua file", "*.*")])
        for path in paths:
            fname = os.path.basename(path)
            stem  = os.path.splitext(fname)[0]
            if stem not in self.gt_dict:
                try:
                    with open(path, encoding="utf-8") as f:
                        content = f.read()
                except Exception:
                    with open(path, encoding="latin-1") as f:
                        content = f.read()
                self.gt_dict[stem] = content
                self.gt_files.append((stem, fname))
                self.lb_gt.insert("end", fname)
        self._update_status()

    def _hapus_gambar(self):
        sel = list(self.lb_gambar.curselection())
        for i in reversed(sel):
            self.lb_gambar.delete(i)
            self.gambar_list.pop(i)
        self._update_status()

    def _hapus_gt(self):
        sel = list(self.lb_gt.curselection())
        for i in reversed(sel):
            self.lb_gt.delete(i)
            stem, _ = self.gt_files[i]
            self.gt_dict.pop(stem, None)
            self.gt_files.pop(i)
        self._update_status()

    def _update_status(self):
        n_g  = len(self.gambar_list)
        n_gt = len(self.gt_dict)
        self._lbl_g_count.configure(text=f"({n_g})")
        self._lbl_gt_count.configure(text=f"({n_gt})")

        if n_g == 0:
            self._lbl_status.configure(text="Belum ada gambar dipilih.", fg=TEKS2)
            return
        cocok    = sum(1 for _, fn in self.gambar_list
                       if os.path.splitext(fn)[0] in self.gt_dict)
        tanpa_gt = n_g - cocok
        if n_gt == 0:
            self._lbl_status.configure(
                text=f"{n_g} gambar dipilih. Tidak ada teks GT.", fg=KUNING)
        elif tanpa_gt == 0:
            self._lbl_status.configure(
                text=f"✓ {n_g} gambar, semua cocok dengan GT", fg=HIJAU)
        else:
            self._lbl_status.configure(
                text=f"{n_g} gambar: {cocok} cocok GT, {tanpa_gt} tanpa GT", fg=KUNING)

    def _jalankan(self):
        if not self.gambar_list:
            messagebox.showinfo("Info", "Pilih minimal satu gambar terlebih dahulu.")
            return
        if self._running:
            return
        self._running  = True
        self._csv_data = None
        self._btn_csv.configure(state="disabled", bg=TEKS3)
        self._btn_jalankan.configure(state="disabled", bg=TEKS3)
        self._lbl_run_status.configure(text="⏳ Memproses...", fg=KUNING)

        self._queue = queue.Queue()
        threading.Thread(target=self._run_thread, daemon=True).start()
        self._poll_queue()

    def _run_thread(self):
        q    = self._queue
        n_g  = len(self.gambar_list)
        n_t  = len(EVAL_TEKNIK)
        LINE = "─" * 65

        q.put(("log", "\n" + "═" * 65 + "\n", "abu"))
        q.put(("log", "  SISTEM EVALUASI PREPROCESSING OCR\n", "header"))
        q.put(("log", "═" * 65 + "\n", "abu"))
        q.put(("log", f"  Jumlah gambar   : {n_g}\n", "normal"))
        q.put(("log", f"  Jumlah teknik   : {n_t}\n", "normal"))
        q.put(("log", f"  Total pengujian : {n_g * n_t}\n", "normal"))
        if not JIWER_OK:
            q.put(("log", "  [!] jiwer tidak terinstal — CER/WER tidak dihitung\n", "kuning"))
        q.put(("log", "═" * 65 + "\n\n", "abu"))

        CSV_HEADER = [
            "nama_file", "teknik", "cer_persen", "wer_persen",
            "jumlah_karakter_ocr", "jumlah_karakter_gt",
            "jumlah_kata_ocr", "jumlah_kata_gt", "waktu_detik",
        ]
        semua_hasil    = []
        cer_per_teknik = defaultdict(list)
        wer_per_teknik = defaultdict(list)

        for idx, (path, nama_file) in enumerate(self.gambar_list, 1):
            stem         = os.path.splitext(nama_file)[0]
            ground_truth = self.gt_dict.get(stem, "")
            gt_norm      = normalize_text(ground_truth)
            gt_wc        = len(gt_norm.split())          if gt_norm else 0
            gt_cc        = len(gt_norm.replace(" ", "")) if gt_norm else 0

            q.put(("log", LINE + "\n", "abu"))
            q.put(("log", f"[{idx}/{n_g}] ", "abu"))
            q.put(("log", f"{nama_file}", "aksen"))
            gt_info = f"  (GT: {gt_wc} kata)" if ground_truth else "  (tanpa GT)"
            q.put(("log", gt_info + "\n", "abu"))
            q.put(("log", LINE + "\n", "abu"))

            image_bgr = cv2.imread(path)
            if image_bgr is None:
                q.put(("log", "  [ERROR] Gagal membaca file gambar!\n\n", "merah"))
                continue
            image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

            for teknik in EVAL_TEKNIK:
                label = f"  {teknik:<40}"
                q.put(("log", label, "normal"))
                mulai = time.time()
                try:
                    processed = process_single_technique(image, teknik)
                    ocr_text  = run_ocr(processed)
                    ocr_wc    = len([w for w in ocr_text.split() if w.strip()])
                    ocr_cc    = len(normalize_text(ocr_text).replace(" ", ""))

                    if ground_truth:
                        m       = hitung_cer_wer(ocr_text, ground_truth)
                        cer_out = m["cer"] if m["cer"] is not None else -1.0
                        wer_out = m["wer"] if m["wer"] is not None else -1.0
                    else:
                        cer_out = wer_out = -1.0

                    elapsed = round(time.time() - mulai, 2)

                    if cer_out >= 0:
                        def _ctag(v): return "hijau" if v < 20 else ("kuning" if v < 50 else "merah")
                        q.put(("log", "CER: ", "normal"))
                        q.put(("log", f"{cer_out:6.2f}%", _ctag(cer_out)))
                        q.put(("log", "  WER: ", "normal"))
                        q.put(("log", f"{wer_out:6.2f}%", _ctag(wer_out)))
                        q.put(("log", f"  [{elapsed}s]\n", "abu"))
                        cer_per_teknik[teknik].append(cer_out)
                        wer_per_teknik[teknik].append(wer_out)
                    else:
                        q.put(("log", f"(tanpa GT)  [{elapsed}s]\n", "abu"))

                    semua_hasil.append({
                        "nama_file": nama_file, "teknik": teknik,
                        "cer_persen": cer_out, "wer_persen": wer_out,
                        "jumlah_karakter_ocr": ocr_cc, "jumlah_karakter_gt": gt_cc,
                        "jumlah_kata_ocr": ocr_wc, "jumlah_kata_gt": gt_wc,
                        "waktu_detik": elapsed,
                    })

                except Exception as exc:
                    elapsed = round(time.time() - mulai, 2)
                    q.put(("log", f"ERROR: {exc}\n", "merah"))
                    semua_hasil.append({
                        "nama_file": nama_file, "teknik": teknik,
                        "cer_persen": -1.0, "wer_persen": -1.0,
                        "jumlah_karakter_ocr": 0, "jumlah_karakter_gt": gt_cc,
                        "jumlah_kata_ocr": 0, "jumlah_kata_gt": gt_wc,
                        "waktu_detik": elapsed,
                    })

            q.put(("log", "\n", "normal"))

        q.put(("log", "\n" + "═" * 65 + "\n", "abu"))
        q.put(("log", "  RINGKASAN RATA-RATA PER TEKNIK\n", "header"))
        q.put(("log", "═" * 65 + "\n", "abu"))
        q.put(("log", f"  {'TEKNIK':<40} {'Rata CER':>10} {'Rata WER':>10}\n", "normal"))
        q.put(("log", "  " + "─" * 61 + "\n", "abu"))

        for teknik in EVAL_TEKNIK:
            if cer_per_teknik[teknik]:
                avg_c = round(sum(cer_per_teknik[teknik]) / len(cer_per_teknik[teknik]), 2)
                avg_w = round(sum(wer_per_teknik[teknik]) / len(wer_per_teknik[teknik]), 2)
                def _ctag2(v): return "hijau" if v < 20 else ("kuning" if v < 50 else "merah")
                q.put(("log", f"  {teknik:<40} ", "normal"))
                q.put(("log", f"{avg_c:9.2f}%", _ctag2(avg_c)))
                q.put(("log", f" {avg_w:9.2f}%\n", "normal"))
            else:
                q.put(("log", f"  {teknik:<40} {'N/A':>10} {'N/A':>10}\n", "abu"))

        q.put(("log", "═" * 65 + "\n\n", "abu"))
        q.put(("log", "  ✔ SELESAI — Klik [⬇ Simpan CSV] untuk menyimpan hasil.\n", "sukses"))
        q.put(("log", "═" * 65 + "\n", "abu"))

        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=CSV_HEADER)
        writer.writeheader()
        writer.writerows(semua_hasil)
        q.put(("done", buf.getvalue()))

    def _poll_queue(self):
        try:
            while True:
                item = self._queue.get_nowait()
                if item[0] == "log":
                    tag = item[2] if len(item) > 2 else "normal"
                    self._append_terminal(item[1], tag)
                elif item[0] == "done":
                    self._csv_data = item[1]
                    self._on_done()
                    return
        except queue.Empty:
            pass
        if self._running:
            self.after(50, self._poll_queue)

    def _on_done(self):
        self._running = False
        self._btn_jalankan.configure(state="normal", bg=AKSEN)
        self._btn_csv.configure(state="normal", bg=HIJAU)
        self._lbl_run_status.configure(text="✔ Selesai", fg=HIJAU)

    def _simpan_csv(self):
        if not self._csv_data:
            messagebox.showinfo("Info", "Jalankan evaluasi terlebih dahulu.")
            return
        path = filedialog.asksaveasfilename(
            title="Simpan Hasil Evaluasi",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("Semua file", "*.*")],
            initialfile="hasil_evaluasi.csv")
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8-sig", newline="") as f:
                f.write(self._csv_data)
            messagebox.showinfo("Berhasil", f"CSV berhasil disimpan:\n{path}")
        except Exception as exc:
            messagebox.showerror("Error", f"Gagal menyimpan:\n{exc}")

    def _reset(self):
        if self._running:
            messagebox.showinfo("Info", "Evaluasi sedang berjalan, tunggu hingga selesai.")
            return
        self.gambar_list.clear()
        self.gt_dict.clear()
        self.gt_files.clear()
        self.lb_gambar.delete(0, "end")
        self.lb_gt.delete(0, "end")
        self._csv_data = None
        self._btn_csv.configure(state="disabled", bg=TEKS3)
        self._lbl_run_status.configure(text="")
        self._clear_terminal_ui()
        self._update_status()


class RasterisasiFrame(tk.Frame):

    DPI = 300

    def __init__(self, parent, **kw):
        super().__init__(parent, bg=BG, **kw)
        self.pdf_paths    = []
        self._queue       = queue.Queue()
        self._running     = False
        self._hasil_images = []
        self._bangun_ui()

    def _bangun_ui(self):
        self._bangun_topbar()

        content = tk.Frame(self, bg=BG)
        content.pack(fill="both", expand=True)

        self.frame_atas = tk.Frame(content, bg=BG, height=265)
        self.frame_atas.pack(fill="x", padx=12, pady=(0, 4))
        self.frame_atas.pack_propagate(False)
        self.frame_atas.columnconfigure(0, weight=2, uniform="atas", minsize=200)
        self.frame_atas.columnconfigure(1, weight=1, uniform="atas", minsize=160)
        self.frame_atas.rowconfigure(0, weight=1, minsize=265)

        self._bangun_card_pdf(self.frame_atas)
        self._bangun_card_pengaturan(self.frame_atas)

        self.frame_bawah = tk.Frame(content, bg=BG)
        self.frame_bawah.pack(fill="both", expand=True, padx=12, pady=(0, 8))

        self._bangun_card_log(self.frame_bawah)

    def _bangun_card_pdf(self, parent):
        f = tk.Frame(parent, bg=PANEL)
        f.grid(row=0, column=0, sticky="nsew", padx=(0, 4), pady=4)

        hdr = tk.Frame(f, bg=PANEL)
        hdr.pack(fill="x", padx=12, pady=(10, 6))
        tk.Frame(hdr, bg=KUNING, width=3, height=16).pack(side="left", padx=(0, 8))
        tk.Label(hdr, text="File PDF", font=F_JUDUL, bg=PANEL, fg=PUTIH).pack(side="left")
        self._lbl_count = tk.Label(hdr, text="(0)", font=F_KECIL, bg=PANEL, fg=TEKS2)
        self._lbl_count.pack(side="left", padx=4)

        frame_lb = tk.Frame(f, bg=CARD)
        frame_lb.pack(fill="both", expand=True, padx=12, pady=(0, 4))
        sb = ttk.Scrollbar(frame_lb, orient="vertical")
        sb.pack(side="right", fill="y", pady=4)
        self.lb_pdf = tk.Listbox(
            frame_lb, font=F_KECIL, bg=CARD, fg=TEKS,
            selectbackground="#a07800", selectforeground=PUTIH,
            relief="flat", bd=0, activestyle="none",
            selectmode="extended", yscrollcommand=sb.set)
        self.lb_pdf.pack(side="left", fill="both", expand=True, padx=(6, 0), pady=4)
        sb.configure(command=self.lb_pdf.yview)

        btn_f = tk.Frame(f, bg=PANEL)
        btn_f.pack(fill="x", padx=12, pady=(0, 10))
        buat_tombol(btn_f, "+ Tambah", self._pilih_pdf,
                    warna="#21262d", lebar=10).pack(side="left", padx=(0, 4))
        buat_tombol(btn_f, "− Hapus", self._hapus_pdf,
                    warna="#21262d", lebar=10).pack(side="left")

    def _bangun_card_pengaturan(self, parent):
        f = tk.Frame(parent, bg=PANEL)
        f.grid(row=0, column=1, sticky="nsew", padx=(4, 0), pady=4)

        hdr = tk.Frame(f, bg=PANEL)
        hdr.pack(fill="x", padx=12, pady=(10, 6))
        tk.Frame(hdr, bg=AKSEN, width=3, height=16).pack(side="left", padx=(0, 8))
        tk.Label(hdr, text="Pengaturan Konversi", font=F_JUDUL, bg=PANEL, fg=PUTIH).pack(side="left")

        tk.Frame(f, bg=BORDER, height=1).pack(fill="x", padx=12, pady=(0, 10))

        info = [
            f"  • DPI       : {self.DPI}",
            "  • Format    : PNG",
            "  • Penamaan  : nama_page1.png",
            "  • Warna     : RGB (3 kanal)",
        ]
        for line in info:
            tk.Label(f, text=line, font=F_KECIL, bg=PANEL, fg=TEKS2, anchor="w"
                     ).pack(padx=12, fill="x", pady=2)

        if not FITZ_OK:
            tk.Frame(f, bg=BORDER, height=1).pack(fill="x", padx=12, pady=(12, 6))
            tk.Label(f, text="⚠  PyMuPDF tidak terinstal!",
                     font=F_BADGE, bg=PANEL, fg=MERAH, anchor="w"
                     ).pack(padx=12, fill="x")
            tk.Label(f, text="pip install pymupdf",
                     font=F_MONO, bg=CARD, fg=KUNING, anchor="w", padx=8, pady=4
                     ).pack(padx=12, fill="x", pady=(4, 0))

    def _bangun_card_log(self, parent):
        f = tk.Frame(parent, bg=PANEL)
        f.pack(fill="both", expand=True, pady=4)

        hdr = tk.Frame(f, bg=PANEL)
        hdr.pack(fill="x", padx=12, pady=(10, 6))
        tk.Frame(hdr, bg=KUNING, width=3, height=16).pack(side="left", padx=(0, 8))
        tk.Label(hdr, text="Log Konversi", font=F_JUDUL, bg=PANEL, fg=PUTIH).pack(side="left")

        tk.Frame(f, bg=BORDER, height=1).pack(fill="x")

        term_f = tk.Frame(f, bg=TERM_BG)
        term_f.pack(fill="both", expand=True)

        self.terminal = tk.Text(
            term_f, font=F_TERM, bg=TERM_BG, fg=TEKS,
            insertbackground=TEKS, relief="flat", bd=0,
            wrap="none", padx=14, pady=12, state="disabled")
        sb_y = ttk.Scrollbar(term_f, orient="vertical",   command=self.terminal.yview)
        sb_x = ttk.Scrollbar(term_f, orient="horizontal", command=self.terminal.xview)
        self.terminal.configure(yscrollcommand=sb_y.set, xscrollcommand=sb_x.set)
        sb_y.pack(side="right",  fill="y")
        sb_x.pack(side="bottom", fill="x")
        self.terminal.pack(fill="both", expand=True)

        self.terminal.tag_configure("normal",  foreground=TEKS)
        self.terminal.tag_configure("header",  foreground=KUNING, font=("Consolas", 9, "bold"))
        self.terminal.tag_configure("aksen",   foreground=AKSEN,  font=("Consolas", 9, "bold"))
        self.terminal.tag_configure("hijau",   foreground=HIJAU)
        self.terminal.tag_configure("merah",   foreground=MERAH)
        self.terminal.tag_configure("kuning",  foreground=KUNING)
        self.terminal.tag_configure("abu",     foreground=TEKS3)
        self.terminal.tag_configure("sukses",  foreground=HIJAU, font=("Consolas", 9, "bold"))
        self.terminal.tag_configure("bold",    font=("Consolas", 9, "bold"))

    def _bangun_topbar(self):
        bar = tk.Frame(self, bg=PANEL, height=58)
        bar.pack(fill="x")
        bar.pack_propagate(False)

        kiri = tk.Frame(bar, bg=PANEL)
        kiri.pack(side="left", padx=20, pady=10)
        tk.Frame(kiri, bg=KUNING, width=4, height=30).pack(side="left", padx=(0, 10))
        tk.Label(kiri, text="Rasterisasi PDF ke Gambar",
                 font=("Segoe UI", 13, "bold"), bg=PANEL, fg=PUTIH).pack(side="left")

        kanan = tk.Frame(bar, bg=PANEL)
        kanan.pack(side="right", padx=16)

        self._btn_konversi = buat_tombol(kanan, "▶  Konversi",
                                          self._konversi, warna=AKSEN, lebar=14)
        self._btn_konversi.pack(side="left", padx=4)
        self._btn_simpan = buat_tombol(kanan, "⬇  Simpan PNG",
                                        self._simpan_png, warna=TEKS3, lebar=16)
        self._btn_simpan.pack(side="left", padx=4)
        self._btn_simpan.configure(state="disabled")
        buat_tombol(kanan, "🗑  Bersihkan",
                    self._clear_terminal_ui, warna="#21262d", lebar=14).pack(side="left", padx=4)
        buat_tombol(kanan, "↺  Reset",
                    self._reset, warna="#21262d", lebar=10).pack(side="left", padx=4)

        self._lbl_run_status = tk.Label(bar, text="", font=F_KECIL, bg=PANEL, fg=TEKS2)
        self._lbl_run_status.pack(side="right", padx=12)

    def _append_terminal(self, text: str, tag: str = "normal"):
        self.terminal.configure(state="normal")
        self.terminal.insert("end", text, tag)
        self.terminal.see("end")
        self.terminal.configure(state="disabled")

    def _clear_terminal_ui(self):
        self.terminal.configure(state="normal")
        self.terminal.delete("1.0", "end")
        self.terminal.configure(state="disabled")

    def _pilih_pdf(self):
        paths = filedialog.askopenfilenames(
            title="Pilih File PDF",
            filetypes=[("PDF Files", "*.pdf"), ("Semua file", "*.*")])
        added = 0
        for path in paths:
            fname = os.path.basename(path)
            if not any(p == path for p, _ in self.pdf_paths):
                self.pdf_paths.append((path, fname))
                self.lb_pdf.insert("end", fname)
                added += 1
        if added:
            self._lbl_count.configure(text=f"({len(self.pdf_paths)})")

    def _hapus_pdf(self):
        sel = list(self.lb_pdf.curselection())
        for i in reversed(sel):
            self.lb_pdf.delete(i)
            self.pdf_paths.pop(i)
        self._lbl_count.configure(text=f"({len(self.pdf_paths)})")

    def _konversi(self):
        if not self.pdf_paths:
            messagebox.showinfo("Info", "Pilih minimal satu file PDF terlebih dahulu.")
            return
        if not FITZ_OK:
            messagebox.showerror(
                "Error",
                "PyMuPDF (fitz) tidak terinstal.\nJalankan: pip install pymupdf")
            return
        if self._running:
            return

        self._running = True
        self._btn_konversi.configure(state="disabled", bg=TEKS3)
        self._lbl_run_status.configure(text="⏳ Mengkonversi...", fg=KUNING)

        self._queue = queue.Queue()
        threading.Thread(target=self._run_thread, daemon=True).start()
        self._poll_queue()

    def _run_thread(self):
        q          = self._queue
        total_pdf  = len(self.pdf_paths)
        total_png  = 0
        hasil_images = []
        LINE       = "─" * 55

        q.put(("log", "\n" + "═" * 55 + "\n", "abu"))
        q.put(("log", "  RASTERISASI PDF KE GAMBAR\n", "header"))
        q.put(("log", "═" * 55 + "\n", "abu"))
        q.put(("log", f"  Jumlah PDF    : {total_pdf}\n", "normal"))
        q.put(("log", f"  DPI           : {self.DPI}\n", "normal"))
        q.put(("log", "  Catatan       : hasil disimpan di memori,\n", "normal"))
        q.put(("log", "                  belum ditulis ke disk\n", "normal"))
        q.put(("log", "═" * 55 + "\n\n", "abu"))

        for idx, (path, fname) in enumerate(self.pdf_paths, 1):
            q.put(("log", LINE + "\n", "abu"))
            q.put(("log", f"[{idx}/{total_pdf}] ", "abu"))
            q.put(("log", f"{fname}\n", "header"))
            q.put(("log", LINE + "\n", "abu"))

            try:
                doc         = fitz.open(path)
                base        = os.path.splitext(fname)[0]
                total_pages = len(doc)

                for i, page in enumerate(doc):
                    pix  = page.get_pixmap(
                        matrix=fitz.Matrix(self.DPI / 72, self.DPI / 72))
                    data = np.frombuffer(pix.samples, dtype=np.uint8)

                    if pix.n == 1:
                        arr = np.stack(
                            [data.reshape(pix.height, pix.width)] * 3, axis=-1)
                    else:
                        arr = data.reshape(pix.height, pix.width, pix.n)
                        if pix.n == 4:
                            arr = arr[:, :, :3]
                    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

                    out_name = f"{base}_page{i + 1}.png"
                    hasil_images.append((out_name, bgr))
                    total_png += 1

                    q.put(("log", f"  Page {i+1:>3}/{total_pages} → ", "abu"))
                    q.put(("log", f"{out_name}\n", "hijau"))

                doc.close()
                q.put(("log", f"\n  ✔ {total_pages} halaman selesai dirender\n\n", "sukses"))

            except Exception as exc:
                q.put(("log", f"  [ERROR] {exc}\n\n", "merah"))

        q.put(("log", "═" * 55 + "\n", "abu"))
        q.put(("log", f"  SELESAI — {total_png} gambar siap di memori\n", "sukses"))
        q.put(("log", "  Klik [⬇ Simpan PNG] untuk menyimpan ke folder pilihan Anda\n", "normal"))
        q.put(("log", "═" * 55 + "\n", "abu"))
        q.put(("done", total_png, hasil_images))

    def _poll_queue(self):
        try:
            while True:
                item = self._queue.get_nowait()
                if item[0] == "log":
                    tag = item[2] if len(item) > 2 else "normal"
                    self._append_terminal(item[1], tag)
                elif item[0] == "done":
                    self._on_done(item[1], item[2])
                    return
        except queue.Empty:
            pass
        if self._running:
            self.after(50, self._poll_queue)

    def _on_done(self, total_png: int, hasil_images: list):
        self._running      = False
        self._hasil_images = hasil_images
        self._btn_konversi.configure(state="normal", bg=AKSEN)
        self._lbl_run_status.configure(text=f"✔ {total_png} gambar siap", fg=HIJAU)
        if total_png > 0:
            self._btn_simpan.configure(state="normal", bg=HIJAU)
            messagebox.showinfo(
                "Konversi Selesai",
                f"{total_png} gambar berhasil dirender (di memori).\n\n"
                f"Klik [⬇ Simpan PNG] untuk memilih folder tujuan penyimpanan.")

    def _simpan_png(self):
        if not self._hasil_images:
            messagebox.showinfo("Info", "Belum ada hasil konversi.")
            return

        tujuan = filedialog.askdirectory(title="Pilih Folder Tujuan Penyimpanan PNG")
        if not tujuan:
            return

        berhasil = 0
        gagal    = []
        for fname, bgr in self._hasil_images:
            dst = os.path.join(tujuan, fname)
            try:
                ok = cv2.imwrite(dst, bgr)
                if ok:
                    berhasil += 1
                else:
                    gagal.append(f"{fname}: gagal menulis file")
            except Exception as exc:
                gagal.append(f"{fname}: {exc}")

        self._append_terminal(
            f"\n[⬇ SIMPAN PNG] → {tujuan}\n"
            f"  {berhasil} file berhasil disimpan"
            + (f", {len(gagal)} gagal\n" if gagal else "\n"),
            "sukses" if not gagal else "kuning",
        )
        for err in gagal:
            self._append_terminal(f"  [GAGAL] {err}\n", "merah")

        if gagal:
            messagebox.showwarning(
                "Sebagian Gagal",
                f"{berhasil} file berhasil disimpan.\n"
                f"{len(gagal)} file gagal:\n" + "\n".join(gagal[:5]))
        else:
            messagebox.showinfo(
                "Berhasil",
                f"{berhasil} file PNG berhasil disimpan ke:\n{tujuan}")

    def _reset(self):
        if self._running:
            messagebox.showinfo("Info", "Konversi sedang berjalan, tunggu hingga selesai.")
            return
        self.pdf_paths      = []
        self._hasil_images  = []
        self.lb_pdf.delete(0, "end")
        self._lbl_count.configure(text="(0)")
        self._lbl_run_status.configure(text="")
        self._btn_simpan.configure(state="disabled", bg=TEKS3)
        self._clear_terminal_ui()


class VisualisasiFrame(tk.Frame):
    """Menu baru: memvisualisasikan CARA KERJA setiap teknik preprocessing
    langkah demi langkah — perhitungan, angka hasil, dan gambar hasil tiap
    tahap — termasuk keputusan aktivasi CLAHE & Line Removal.
    """

    LANGKAH = [
        ("alur",   "🔀  Alur Pipeline"),
        ("gray",   "1️⃣  Grayscale"),
        ("clahe",  "2️⃣  CLAHE"),
        ("agt",    "3️⃣  Adaptive Gaussian Thresholding"),
        ("line",   "4️⃣  Line Removal"),
        ("cerwer", "5️⃣  CER & WER"),
    ]

    def __init__(self, parent, **kw):
        super().__init__(parent, bg=BG, **kw)
        self.img_asli      = None
        self.gray          = None
        self.nama_file     = ""
        self.langkah_aktif = "alur"
        self.tile_pilih    = (0, 0)
        self.titik_gray    = None
        self.titik_agt     = None
        self._img_refs     = []
        self._nav_btn      = {}
        self._std = self._tiles = self._clahe_asli = self._binary = self._line = None
        self._hasil_akhir  = None
        self.teks_gt_input  = ""
        self.teks_hyp_input = ""
        self._cerwer_hasil  = None
        self._bangun_ui()

    # ---------------------------------------------------- kerangka umum
    def _bangun_ui(self):
        self._bangun_header()
        self._bangun_subnav()
        self._bangun_scroll_container()
        self._tampilkan_langkah("alur")

    def _bangun_header(self):
        hdr = tk.Frame(self, bg=PANEL, height=58)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)

        kiri = tk.Frame(hdr, bg=PANEL)
        kiri.pack(side="left", padx=20, pady=10)
        tk.Frame(kiri, bg=TEAL, width=4, height=30).pack(side="left", padx=(0, 10))
        tk.Label(kiri, text="Visualisasi Proses Preprocessing",
                 font=("Segoe UI", 13, "bold"), bg=PANEL, fg=PUTIH).pack(side="left")

        kanan = tk.Frame(hdr, bg=PANEL)
        kanan.pack(side="right", padx=16)
        buat_tombol(kanan, "📁  Pilih Gambar", self._pilih_gambar,
                    warna="#21262d", lebar=16).pack(side="left", padx=4)
        buat_tombol(kanan, "↺  Reset", self._reset,
                    warna="#21262d", lebar=10).pack(side="left", padx=4)

        self._lbl_file = tk.Label(hdr, text="Belum ada gambar dipilih.",
                                   font=F_KECIL, bg=PANEL, fg=TEKS2)
        self._lbl_file.pack(side="right", padx=(0, 16))

    def _bangun_subnav(self):
        bar = tk.Frame(self, bg=PANEL)
        bar.pack(fill="x")
        inner = tk.Frame(bar, bg=PANEL)
        inner.pack(anchor="w")
        for key, label in self.LANGKAH:
            wrap = tk.Frame(inner, bg=PANEL)
            wrap.pack(side="left")
            btn = tk.Button(wrap, text=label, font=F_LABEL, bg=PANEL, fg=TEKS2,
                             activebackground="#1c2130", activeforeground=PUTIH,
                             relief="flat", bd=0, padx=16, pady=10, cursor="hand2",
                             command=lambda k=key: self._tampilkan_langkah(k))
            btn.pack()
            ind = tk.Frame(wrap, bg=PANEL, height=2)
            ind.pack(fill="x")
            self._nav_btn[key] = (btn, ind)
        tk.Frame(self, bg=BORDER, height=1).pack(fill="x")

    def _bangun_scroll_container(self):
        wrap = tk.Frame(self, bg=BG)
        wrap.pack(fill="both", expand=True)
        sb = ttk.Scrollbar(wrap, orient="vertical")
        sb.pack(side="right", fill="y")
        self._canvas_scroll = tk.Canvas(wrap, bg=BG, highlightthickness=0, yscrollcommand=sb.set)
        self._canvas_scroll.pack(side="left", fill="both", expand=True)
        sb.configure(command=self._canvas_scroll.yview)
        self.konten = tk.Frame(self._canvas_scroll, bg=BG)
        self._win_id = self._canvas_scroll.create_window((0, 0), window=self.konten, anchor="nw")
        self.konten.bind("<Configure>", lambda e: self._canvas_scroll.configure(
            scrollregion=self._canvas_scroll.bbox("all")))
        self._canvas_scroll.bind("<Configure>",
            lambda e: self._canvas_scroll.itemconfigure(self._win_id, width=e.width))
        self._canvas_scroll.bind("<MouseWheel>", self._on_mousewheel)
        self._canvas_scroll.bind("<Button-4>", lambda e: self._canvas_scroll.yview_scroll(-1, "units"))
        self._canvas_scroll.bind("<Button-5>", lambda e: self._canvas_scroll.yview_scroll(1, "units"))
        self.konten.bind("<MouseWheel>", self._on_mousewheel)
        # add="+" -> ditambahkan berdampingan, TIDAK menggantikan binding scroll milik AppFrame
        self.bind_all("<MouseWheel>", self._on_mousewheel, add="+")

    def _on_mousewheel(self, event):
        if self.winfo_ismapped():
            self._canvas_scroll.yview_scroll(int(-1 * (event.delta / 120)), "units")

    # ---------------------------------------------------- navigasi antar langkah
    def _tampilkan_langkah(self, key):
        self.langkah_aktif = key
        for k, (btn, ind) in self._nav_btn.items():
            aktif = (k == key)
            btn.configure(fg=PUTIH if aktif else TEKS2, bg=CARD if aktif else PANEL)
            ind.configure(bg=TEAL if aktif else PANEL)

        for w in self.konten.winfo_children():
            w.destroy()
        self._img_refs.clear()

        if self.img_asli is None:
            self._panel_kosong(self.konten)
            return

        pembangun = {
            "alur":   self._panel_alur,
            "gray":   self._panel_grayscale,
            "clahe":  self._panel_clahe,
            "agt":    self._panel_agt,
            "line":   self._panel_line,
            "cerwer": self._panel_cerwer,
        }[key]
        pembangun(self.konten)

        # Konten di-rebuild dinamis (destroy + create ulang), beda dengan halaman statis
        # lain di aplikasi ini, sehingga scrollregion perlu dipaksa update manual di sini
        # supaya seluruh konten baru tetap bisa dijangkau lewat scroll.
        self.konten.update_idletasks()
        self._canvas_scroll.configure(scrollregion=self._canvas_scroll.bbox("all"))
        self._canvas_scroll.yview_moveto(0)

    def _panel_kosong(self, parent):
        f = tk.Frame(parent, bg=BG)
        f.pack(fill="both", expand=True, pady=90)
        tk.Label(f, text="📁", font=("Segoe UI", 36), bg=BG, fg=TEKS3).pack()
        tk.Label(f, text="Pilih gambar terlebih dahulu untuk melihat visualisasi proses perhitungannya.",
                 font=F_LABEL, bg=BG, fg=TEKS2).pack(pady=(8, 0))

    # ---------------------------------------------------- load gambar & precompute
    def _pilih_gambar(self):
        path = filedialog.askopenfilename(
            filetypes=[("Gambar", "*.png *.jpg *.jpeg *.bmp"), ("Semua file", "*.*")])
        if not path:
            return
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            messagebox.showerror("Error", "Gagal membaca gambar.")
            return
        self.img_asli  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        self.gray      = to_grayscale(self.img_asli)
        self.nama_file = os.path.basename(path)
        h, w = self.gray.shape
        self._lbl_file.configure(text=f"—  {self.nama_file}  ({w}×{h} px)")

        self.titik_gray = None
        self.titik_agt  = None
        self.tile_pilih = (0, 0)
        self.teks_gt_input  = ""
        self.teks_hyp_input = ""
        self._cerwer_hasil  = None
        self._hitung_semua()
        self._tampilkan_langkah(self.langkah_aktif)

    def _reset(self):
        self.img_asli = self.gray = None
        self.nama_file = ""
        self._lbl_file.configure(text="Belum ada gambar dipilih.")
        self.titik_gray = None
        self.titik_agt  = None
        self.tile_pilih = (0, 0)
        self.teks_gt_input  = ""
        self.teks_hyp_input = ""
        self._cerwer_hasil  = None
        self._std = self._tiles = self._clahe_asli = self._binary = self._line = None
        self._hasil_akhir = None
        self._tampilkan_langkah("alur")

    def _hitung_semua(self):
        gray = self.gray
        self._std        = viz_std_deviasi(gray)
        self._tiles       = viz_bagi_tile(gray, CLAHE_TILE_GRID)
        self._clahe_asli  = clahe_paksa(gray)
        blur              = cv2.medianBlur(gray, 3)
        self._binary      = cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
            AGT_BLOCK_SIZE, AGT_C)
        self._line        = viz_line_removal_detail(self._binary)
        self._hasil_akhir, _ = pipeline_adaptive(self.img_asli)

    def _hitung_cerwer(self):
        gt  = normalize_text(self.teks_gt_input)
        hyp = normalize_text(self.teks_hyp_input)
        nc  = len(gt)
        nw  = len(gt.split())

        r_char = viz_levenshtein_detail(list(gt), list(hyp))
        r_kata = viz_levenshtein_detail(gt.split(), hyp.split())
        cer = round(r_char["distance"] / nc * 100, 2) if nc else None
        wer = round(r_kata["distance"] / nw * 100, 2) if nw else None

        self._cerwer_hasil = {
            "gt": gt, "hyp": hyp, "nc": nc, "nw": nw,
            "char": r_char, "kata": r_kata, "cer": cer, "wer": wer,
        }

    # ---------------------------------------------------- helper tampilan
    def _foto(self, arr, w=None, h=None, nearest=False):
        a = arr
        if a.dtype != np.uint8:
            a = np.clip(a, 0, 255).astype(np.uint8)
        pil = Image.fromarray(a).convert("RGB") if len(a.shape) == 2 else Image.fromarray(a)
        if w or h:
            w = w or max(1, int(pil.width * (h / pil.height)))
            h = h or max(1, int(pil.height * (w / pil.width)))
            pil = pil.resize((max(1, w), max(1, h)), Image.NEAREST if nearest else Image.LANCZOS)
        foto = ImageTk.PhotoImage(pil)
        self._img_refs.append(foto)
        return foto

    def _thumb(self, parent, arr, caption, w=140, h=140, nearest=False):
        ah, aw = arr.shape[:2]
        skala = min(w / aw, h / ah)
        fw, fh = max(1, int(aw * skala)), max(1, int(ah * skala))
        f = tk.Frame(parent, bg=PANEL)
        foto = self._foto(arr, w=fw, h=fh, nearest=nearest)
        tk.Label(f, image=foto, bg=PANEL, highlightbackground=BORDER,
                 highlightthickness=1).pack(padx=2, pady=2)
        tk.Label(f, text=caption, font=F_KECIL, bg=PANEL, fg=TEKS2,
                 wraplength=w + 30, justify="center").pack(pady=(2, 8))
        return f

    def _preview_klik(self, parent, arr, on_klik, lebar_maks=380, tinggi_maks=380,
                       titik=None, kotak_ukuran=None):
        h0, w0 = arr.shape[:2]
        skala = min(lebar_maks / w0, tinggi_maks / h0)
        lebar = max(1, int(w0 * skala))
        tinggi = max(1, int(h0 * skala))
        disp = arr.copy()
        if len(disp.shape) == 2:
            disp = cv2.cvtColor(disp, cv2.COLOR_GRAY2RGB)
        if titik is not None:
            cy, cx = titik
            rr = max(3, (kotak_ukuran or 8) // 2)
            tebal = max(2, w0 // 250)
            cv2.rectangle(disp, (cx - rr, cy - rr), (cx + rr, cy + rr), (255, 70, 70), tebal)
        foto = self._foto(disp, w=lebar, h=tinggi)
        lbl = tk.Label(parent, image=foto, bg=PANEL, cursor="crosshair")
        lbl.pack(padx=12, pady=(0, 12))

        def _klik(e):
            ox = max(0, min(w0 - 1, int(e.x / skala)))
            oy = max(0, min(h0 - 1, int(e.y / skala)))
            on_klik(oy, ox)
        lbl.bind("<Button-1>", _klik)
        return lbl

    def _kartu_rumus(self, parent, formula, keterangan=None):
        f = tk.Frame(parent, bg="#161d30", highlightbackground=AKSEN2, highlightthickness=1)
        f.pack(fill="x", padx=16, pady=(0, 10))
        tk.Label(f, text=formula, font=("Segoe UI", 12), bg="#161d30", fg=PUTIH,
                 justify="left", wraplength=920, anchor="w").pack(anchor="w", padx=14, pady=(12, 6))
        if keterangan:
            for k in keterangan:
                tk.Label(f, text=f"·  {k}", font=F_KECIL, bg="#161d30", fg=TEKS2,
                         anchor="w", wraplength=900, justify="left").pack(fill="x", padx=14, pady=1)
            tk.Frame(f, bg="#161d30", height=8).pack()
        return f

    def _badge_keputusan(self, parent, aktif, teks):
        warna = HIJAU if aktif else MERAH
        f = tk.Frame(parent, bg=warna)
        tk.Label(f, text=teks, font=("Segoe UI", 9, "bold"), bg=warna, fg=PUTIH,
                 padx=12, pady=6).pack()
        return f

    def _judul_sub(self, parent, teks):
        tk.Label(parent, text=teks, font=F_JUDUL, bg=BG, fg=PUTIH).pack(anchor="w", padx=16, pady=(4, 6))

    def _garis(self, parent, atas=16, bawah=16):
        tk.Frame(parent, bg=BORDER, height=1).pack(fill="x", padx=16, pady=(atas, bawah))

    # ---------------------------------------------------- helper isi terminal
    def _tulis(self, txt, s, tag="normal"):
        txt.configure(state="normal")
        txt.insert("end", s, tag)
        txt.configure(state="disabled")

    def _isi_terminal_grayscale(self, txt, hasil):
        self._tulis(txt, f"Patch dasar : (x={hasil['x0']}, y={hasil['y0']})  ukuran "
                      f"{hasil['ukuran']}x{hasil['ukuran']} piksel\n\n", "abu")
        self._tulis(txt, f"{'Koord (x,y)':<14}{'R':>4}{'G':>4}{'B':>4}   "
                      f"{'Substitusi':<38}{'Hasil':>7}\n", "header")
        self._tulis(txt, "-" * 84 + "\n", "abu")
        for c in hasil["contoh"]:
            sub = f"0.299x{c['r']}+0.587x{c['g']}+0.114x{c['b']}={c['gray_exact']:.2f}"
            self._tulis(txt, f"({c['x']:>4},{c['y']:>4})   ", "normal")
            self._tulis(txt, f"{c['r']:>3}{c['g']:>4}{c['b']:>4}   ", "aksen")
            self._tulis(txt, f"{sub:<38}", "normal")
            self._tulis(txt, f"-> {c['gray_bulat']:>4}\n", "hijau")

    def _isi_terminal_std(self, txt, std):
        self._tulis(txt, "Dihitung atas SELURUH piksel citra grayscale:\n\n", "abu")
        self._tulis(txt, f"  Jumlah piksel (N)             = {std['n']}\n", "normal")
        self._tulis(txt, f"  Jumlah intensitas (jumlah I)  = {std['jumlah']:.2f}\n", "normal")
        self._tulis(txt, f"  Rata-rata (mu = jumlah I / N) = {std['mean']:.4f}\n", "normal")
        self._tulis(txt, f"  Jumlah (I - mu)^2             = {std['jumlah_sqdiff']:.2f}\n", "normal")
        self._tulis(txt, f"  Variansi (sigma^2)            = {std['variansi']:.4f}\n", "normal")
        self._tulis(txt, f"  Standar deviasi (sigma)       = {std['std']:.4f}\n", "aksen")
        self._tulis(txt, f"  Threshold                     = {std['threshold']:.0f}\n\n", "normal")
        if std["aktif"]:
            self._tulis(txt, f"  sigma ({std['std']:.2f}) < {std['threshold']:.0f}   ->   ", "normal")
            self._tulis(txt, "CLAHE DIAKTIFKAN\n", "sukses")
        else:
            self._tulis(txt, f"  sigma ({std['std']:.2f}) >= {std['threshold']:.0f}   ->   ", "normal")
            self._tulis(txt, "CLAHE DILEWATI, lanjut ke Adaptive Gaussian Thresholding\n", "kuning")

    def _isi_terminal_tile(self, txt, i, j, tinfo, hist):
        self._tulis(txt, f"Tile terpilih : baris {i}, kolom {j}  "
                      f"(baris piksel {tinfo['y0']}-{tinfo['y1']}, kolom {tinfo['x0']}-{tinfo['x1']})\n\n", "abu")
        self._tulis(txt, f"  N_tile (jumlah piksel tile) = {hist['n_tile']}\n", "normal")
        self._tulis(txt, f"  L (jumlah level intensitas) = {hist['L']}\n", "normal")
        self._tulis(txt, f"  Tc (clip limit)             = {hist['Tc']}\n\n", "normal")
        contoh_k = [0, 32, 64, 96, 128, 160, 192, 224, 255]
        self._tulis(txt, "  1) Clipping histogram:\n", "header")
        for k in contoh_k:
            self._tulis(txt, f"    H({k:>3})={int(hist['hist'][k]):>4}  ->  H_clip({k:>3})="
                          f"min({int(hist['hist'][k])},{hist['Tc']})={hist['hist_clip'][k]:.1f}\n", "normal")
        kelebihan = hist["sum_hist"] - hist["sum_hist_clip"]
        self._tulis(txt, f"\n  jumlah piksel yang terpotong = {hist['sum_hist']:.0f} - "
                      f"{hist['sum_hist_clip']:.0f} = {kelebihan:.0f}\n\n", "abu")
        self._tulis(txt, "  2) Redistribusi merata ke semua level:\n", "header")
        self._tulis(txt, f"    H_final(k) = H_clip(k) + {kelebihan:.0f}/{hist['L']} "
                      f"= H_clip(k) + {kelebihan / hist['L']:.3f}\n\n", "normal")
        self._tulis(txt, "  3) Pemetaan s_k:\n", "header")
        for k in contoh_k:
            self._tulis(txt, f"    s_{k:<3} = 255 x {hist['kumulatif'][k]:.1f} / {hist['n_tile']} = "
                          f"{hist['s_k'][k]:.3f}  ->  dibulatkan {int(hist['lut'][k])}\n", "normal")

    def _isi_terminal_agt(self, txt, r):
        cy, cx = r["titik"]
        self._tulis(txt, f"Titik dihitung : (x={cx}, y={cy})   window {r['block_size']}x{r['block_size']} "
                      f"piksel, C={r['C']}\n", "abu")
        self._tulis(txt, f"Sigma Gaussian : sigma = 0.3x((blockSize-1)x0.5-1)+0.8 = {r['sigma']:.4f}  "
                      f"(formula standar)\n\n", "abu")
        self._tulis(txt, f"  jumlah bobot w(i,j)            = {r['sum_bobot']:.6f}\n", "normal")
        self._tulis(txt, f"  jumlah w(i,j) x I_blur(i,j)    = {r['sum_bobot_i']:.4f}\n", "normal")
        self._tulis(txt, f"  T(x,y) = {r['sum_bobot_i']:.4f} / {r['sum_bobot']:.6f} - {r['C']} = ", "normal")
        self._tulis(txt, f"{r['T']:.4f}\n\n", "aksen")
        self._tulis(txt, f"  I(x,y) piksel pusat (setelah median blur 3x3) = {r['pusat_blur']:.0f}\n", "normal")
        if r["hasil"] == 255:
            self._tulis(txt, f"  {r['pusat_blur']:.0f} > {r['T']:.4f}   ->   ", "normal")
            self._tulis(txt, "piksel = 255 (putih)\n", "sukses")
        else:
            self._tulis(txt, f"  {r['pusat_blur']:.0f} <= {r['T']:.4f}   ->   ", "normal")
            self._tulis(txt, "piksel = 0 (hitam)\n", "sukses")

    def _isi_terminal_line(self, txt, r):
        self._tulis(txt, f"Kernel horizontal (K_h) = {r['kh_size'][0]}x{r['kh_size'][1]} piksel  "
                      f"(maks(10, lebar/40))\n", "abu")
        self._tulis(txt, f"Kernel vertikal   (K_v) = {r['kv_size'][0]}x{r['kv_size'][1]} piksel  "
                      f"(maks(10, tinggi/40))\n\n", "abu")
        self._tulis(txt, f"  P_garis (piksel pada mask) = {r['p_garis']}\n", "normal")
        self._tulis(txt, f"  P_total (total piksel)     = {r['p_total']}\n", "normal")
        self._tulis(txt, f"  Rg = P_garis/P_total = {r['p_garis']}/{r['p_total']} = {r['rg']:.5f}\n", "aksen")
        self._tulis(txt, f"  Threshold                  = {r['threshold']}\n\n", "normal")
        if r["aktif"]:
            self._tulis(txt, f"  Rg ({r['rg']:.5f}) > {r['threshold']}   ->   ", "normal")
            self._tulis(txt, "LINE REMOVAL DIJALANKAN\n", "sukses")
        else:
            self._tulis(txt, f"  Rg ({r['rg']:.5f}) <= {r['threshold']}   ->   ", "normal")
            self._tulis(txt, "LINE REMOVAL DILEWATI, citra biner dipakai apa adanya\n", "kuning")

    # ---------------------------------------------------- callback interaktif
    def _klik_titik_gray(self, y, x):
        self.titik_gray = (y, x)
        self._tampilkan_langkah("gray")

    def _klik_titik_agt(self, y, x):
        self.titik_agt = (y, x)
        self._tampilkan_langkah("agt")

    def _pilih_tile(self, i, j):
        self.tile_pilih = (i, j)
        self._tampilkan_langkah("clahe")

    # ---------------------------------------------------- panel: alur pipeline
    def _panel_alur(self, parent):
        judul_seksi(parent, "Alur Pipeline Preprocessing Adaptif", warna_aksen=TEAL)
        tk.Label(parent, text="Klik salah satu kotak untuk melompat ke detail perhitungannya.",
                 font=F_KECIL, bg=BG, fg=TEKS2).pack(anchor="w", padx=16, pady=(0, 10))

        baris = tk.Frame(parent, bg=BG)
        baris.pack(fill="x", padx=16, pady=(0, 20))

        kiri = tk.Frame(baris, bg=BG)
        kiri.pack(side="left", anchor="n")

        cv_ = tk.Canvas(kiri, bg=BG, highlightthickness=0, width=560, height=260)
        cv_.pack(anchor="w")

        std, ln = self._std, self._line

        def kotak(x, y, w, h, judul, sub=None, warna_tepi=BORDER, dash=None, tag=None):
            cv_.create_rectangle(x, y, x + w, y + h, fill=CARD, outline=warna_tepi,
                                  width=2, dash=dash, tags=tag)
            cv_.create_text(x + w / 2, y + (h / 2 - 9 if sub else h / 2), text=judul,
                             font=("Segoe UI", 9, "bold"), fill=PUTIH, width=w - 14,
                             justify="center", tags=tag)
            if sub:
                cv_.create_text(x + w / 2, y + h - 14, text=sub, font=("Segoe UI", 8),
                                 fill=TEKS2, width=w - 12, justify="center", tags=tag)

        def panah(x0, y0, x1, y1):
            cv_.create_line(x0, y0, x1, y1, fill=TEKS3, width=2, arrow="last", arrowshape=(9, 11, 4))

        bw, bh, gap = 150, 76, 34
        xs = [12, 12 + bw + gap, 12 + 2 * (bw + gap)]
        y1_ = 14

        kotak(xs[0], y1_, bw, bh, "Citra Asli\n(RGB)", warna_tepi=AKSEN, tag="b_asli")
        panah(xs[0] + bw, y1_ + bh / 2, xs[1], y1_ + bh / 2)
        kotak(xs[1], y1_, bw, bh, "1. Grayscale", "selalu dijalankan", warna_tepi=AKSEN, tag="b_gray")
        panah(xs[1] + bw, y1_ + bh / 2, xs[2], y1_ + bh / 2)

        c_warna = HIJAU if std["aktif"] else TEKS3
        c_sub = (f"sigma={std['std']:.2f} < {std['threshold']:.0f} -> AKTIF" if std["aktif"]
                  else f"sigma={std['std']:.2f} >= {std['threshold']:.0f} -> DILEWATI")
        kotak(xs[2], y1_, bw, bh, "2. CLAHE", c_sub, warna_tepi=c_warna,
              dash=None if std["aktif"] else (5, 3), tag="b_clahe")

        y2_ = y1_ + bh + 56
        panah(xs[2] + bw / 2, y1_ + bh, xs[2] + bw / 2, y1_ + bh + 24)
        cv_.create_line(xs[2] + bw / 2, y1_ + bh + 24, xs[0] + bw / 2, y1_ + bh + 24,
                         fill=TEKS3, width=2)
        panah(xs[0] + bw / 2, y1_ + bh + 24, xs[0] + bw / 2, y2_)

        kotak(xs[0], y2_, bw, bh, "3. Adaptive Gaussian\nThresholding", "selalu dijalankan",
              warna_tepi=AKSEN, tag="b_agt")
        panah(xs[0] + bw, y2_ + bh / 2, xs[1], y2_ + bh / 2)

        l_warna = HIJAU if ln["aktif"] else TEKS3
        l_sub = (f"Rg={ln['rg']:.4f} > {ln['threshold']} -> AKTIF" if ln["aktif"]
                  else f"Rg={ln['rg']:.4f} <= {ln['threshold']} -> DILEWATI")
        kotak(xs[1], y2_, bw, bh, "4. Line Removal", l_sub, warna_tepi=l_warna,
              dash=None if ln["aktif"] else (5, 3), tag="b_line")
        panah(xs[1] + bw, y2_ + bh / 2, xs[2], y2_ + bh / 2)
        kotak(xs[2], y2_, bw, bh, "Hasil Akhir", warna_tepi=HIJAU, tag="b_hasil")

        for tag, key in [("b_gray", "gray"), ("b_clahe", "clahe"), ("b_agt", "agt"), ("b_line", "line")]:
            cv_.tag_bind(tag, "<Button-1>", lambda e, k=key: self._tampilkan_langkah(k))
            cv_.tag_bind(tag, "<Enter>", lambda e: cv_.configure(cursor="hand2"))
            cv_.tag_bind(tag, "<Leave>", lambda e: cv_.configure(cursor="arrow"))

        legenda = tk.Frame(kiri, bg=BG)
        legenda.pack(anchor="w", pady=(6, 0))
        tk.Frame(legenda, bg=HIJAU, width=14, height=14).pack(side="left")
        tk.Label(legenda, text=" hijau = teknik dijalankan   ",
                 font=F_KECIL, bg=BG, fg=TEKS2).pack(side="left")
        tk.Frame(legenda, bg=TEKS3, width=14, height=14).pack(side="left")
        tk.Label(legenda, text=" abu putus-putus = teknik dilewati",
                 font=F_KECIL, bg=BG, fg=TEKS2).pack(side="left")

        kanan = tk.Frame(baris, bg=BG)
        kanan.pack(side="left", anchor="n", padx=(28, 0), fill="y")
        tk.Label(kanan, text="Perbandingan Citra Sebelum dan Sesudah Preprocessing Adaptif",
                 font=F_JUDUL, bg=BG, fg=PUTIH, wraplength=420, justify="left"
                 ).pack(anchor="w", pady=(0, 12))
        gambar_row = tk.Frame(kanan, bg=BG)
        gambar_row.pack(anchor="w")
        self._thumb(gambar_row, self.img_asli, "Sebelum (Citra Asli)", w=200, h=200
                    ).pack(side="left", padx=(0, 12))
        self._thumb(gambar_row, self._hasil_akhir, "Sesudah (Preprocessing Adaptif)", w=200, h=200
                    ).pack(side="left")

    # ---------------------------------------------------- panel: grayscale
    def _panel_grayscale(self, parent):
        judul_seksi(parent, "1. Grayscale — Konversi RGB ke Skala Abu-abu", warna_aksen=AKSEN)
        self._kartu_rumus(parent,
            "I(x,y) = 0,299 x R(x,y) + 0,587 x G(x,y) + 0,114 x B(x,y)",
            ["I(x,y) = nilai keabuan pada koordinat (x,y)",
             "R(x,y), G(x,y), B(x,y) = nilai kanal merah, hijau, biru pada koordinat yang sama"])

        hasil = viz_grayscale_sample(self.img_asli, titik=self.titik_gray, ukuran=4)

        body = tk.Frame(parent, bg=BG)
        body.pack(fill="x", padx=16)
        body.columnconfigure(0, weight=3)
        body.columnconfigure(1, weight=2)

        kiri = tk.Frame(body, bg=PANEL)
        kiri.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        tk.Label(kiri, text="Klik gambar untuk memindahkan titik contoh perhitungan:",
                 font=F_KECIL, bg=PANEL, fg=TEKS2).pack(anchor="w", padx=12, pady=(12, 4))
        self._preview_klik(kiri, self.img_asli, self._klik_titik_gray, lebar_maks=380, tinggi_maks=340,
                            titik=hasil["titik"], kotak_ukuran=hasil["ukuran"] * 6)

        kanan = tk.Frame(body, bg=PANEL)
        kanan.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        tk.Label(kanan, text=f"Patch {hasil['ukuran']}x{hasil['ukuran']} piksel (diperbesar)",
                 font=F_KECIL, bg=PANEL, fg=TEKS2).pack(anchor="w", padx=12, pady=(12, 6))
        pasangan = tk.Frame(kanan, bg=PANEL)
        pasangan.pack(padx=12, pady=(0, 16))
        self._thumb(pasangan, hasil["patch_asli"], "Asli (RGB)", w=130, h=130, nearest=True).pack(
            side="left", padx=(0, 8))
        self._thumb(pasangan, hasil["patch_gray"], "Grayscale", w=130, h=130, nearest=True).pack(side="left")

        txt = _buat_terminal_panel(parent, warna_aksen=AKSEN, tinggi=20,
            judul=f"Perhitungan {hasil['ukuran']}x{hasil['ukuran']} Piksel pada Titik "
                  f"(x={hasil['titik'][1]}, y={hasil['titik'][0]})")
        self._isi_terminal_grayscale(txt, hasil)
        tk.Frame(parent, bg=BG, height=16).pack()

    # ---------------------------------------------------- panel: CLAHE
    def _panel_clahe(self, parent):
        judul_seksi(parent, "2. CLAHE — Contrast Limited Adaptive Histogram Equalization",
                    warna_aksen=AKSEN2)

        self._judul_sub(parent, "A.  Keputusan Aktivasi (berdasarkan standar deviasi citra)")
        self._kartu_rumus(parent, "sigma = akar( jumlah(I - mu)^2 / N ),   mu = jumlah I / N",
            ["sigma = standar deviasi intensitas seluruh citra grayscale",
             f"CLAHE hanya diaktifkan bila sigma < {CLAHE_STD_THRESHOLD:.0f} (citra dianggap kurang kontras)"])
        txt1 = _buat_terminal_panel(parent, warna_aksen=AKSEN2, tinggi=11,
                                     judul="Perhitungan Standar Deviasi Seluruh Citra")
        self._isi_terminal_std(txt1, self._std)
        badge_row = tk.Frame(parent, bg=BG)
        badge_row.pack(anchor="w", padx=16, pady=(10, 8))
        self._badge_keputusan(badge_row, self._std["aktif"],
            "CLAHE DIAKTIFKAN" if self._std["aktif"]
            else "CLAHE DILEWATI -> lanjut ke Adaptive Gaussian Thresholding").pack(side="left")
        self._garis(parent)

        kolom, baris = CLAHE_TILE_GRID
        self._judul_sub(parent, f"B.  Pembagian Citra menjadi Tile ({baris}x{kolom} = {baris*kolom} tile)")
        tk.Label(parent, text="Setiap tile diproses histogram equalization SENDIRI-SENDIRI. "
                              "Klik salah satu tile untuk melihat detail perhitungannya:",
                 font=F_KECIL, bg=BG, fg=TEKS2, wraplength=920, justify="left"
                 ).pack(anchor="w", padx=16, pady=(0, 10))
        grid_f = tk.Frame(parent, bg=BG)
        grid_f.pack(padx=16, pady=(0, 16), anchor="w")
        for i in range(baris):
            for j in range(kolom):
                tile = self._tiles[i][j]
                terpilih = (i, j) == self.tile_pilih
                th, tw = tile["citra"].shape[:2]
                skala_t = min(78 / tw, 78 / th)
                foto = self._foto(tile["citra"], w=max(1, int(tw * skala_t)), h=max(1, int(th * skala_t)))
                btn = tk.Button(grid_f, image=foto, bg=(TEAL if terpilih else PANEL),
                                bd=3 if terpilih else 1, relief="solid",
                                activebackground=TEAL, cursor="hand2",
                                command=lambda i=i, j=j: self._pilih_tile(i, j))
                btn.grid(row=i * 2, column=j, padx=3, pady=(3, 0))
                tk.Label(grid_f, text=f"({i},{j})", font=F_KECIL, bg=BG,
                         fg=(TEAL if terpilih else TEKS2)).grid(row=i * 2 + 1, column=j, pady=(0, 8))
        self._garis(parent)

        i, j = self.tile_pilih
        tinfo = self._tiles[i][j]
        hist = viz_histogram_tile_manual(tinfo["citra"])
        self._judul_sub(parent, f"C.  Detail Perhitungan Tile ({i},{j})")
        self._kartu_rumus(parent,
            "H_clip(k) = min(H(k), Tc)     H_final(k) = H_clip(k) + kelebihan/L     "
            "s_k = 255 x jumlah[H_final(0..k)] / N_tile",
            ["H(k) = jumlah piksel bernilai k dalam tile;  Tc = clip limit",
             "kelebihan = jumlah piksel yang terpotong saat clipping, disebar rata ke semua level (L level)",
             "s_k = nilai intensitas baru hasil pemetaan untuk level k"])
        perbandingan = tk.Frame(parent, bg=BG)
        perbandingan.pack(padx=16, pady=(0, 10), anchor="w")
        self._thumb(perbandingan, tinfo["citra"], "Tile Sebelum", w=140, h=140).pack(side="left", padx=(0, 10))
        self._thumb(perbandingan, hist["hasil"], "Tile Sesudah", w=140, h=140).pack(side="left")

        txt2 = _buat_terminal_panel(parent, warna_aksen=AKSEN2, tinggi=22,
                                     judul=f"Histogram & Pemetaan Tile ({i},{j})")
        self._isi_terminal_tile(txt2, i, j, tinfo, hist)
        self._garis(parent)

        self._judul_sub(parent, "D.  Hasil Akhir CLAHE (Seluruh Citra)")
        hasil_row = tk.Frame(parent, bg=BG)
        hasil_row.pack(padx=16, pady=(0, 20), anchor="w")
        self._thumb(hasil_row, self.gray, "Sebelum (Grayscale)", w=170, h=170).pack(side="left", padx=(0, 10))
        self._thumb(hasil_row, self._clahe_asli, "Sesudah (CLAHE)", w=170, h=170).pack(side="left")

    # ---------------------------------------------------- panel: AGT
    def _panel_agt(self, parent):
        judul_seksi(parent, "3. Adaptive Gaussian Thresholding", warna_aksen=AKSEN)
        tk.Label(parent, text="Catatan: teknik ini SELALU dijalankan pada seluruh citra tanpa syarat "
                              "aktivasi apa pun (berbeda dari CLAHE & Line Removal).",
                 font=F_KECIL, bg=BG, fg=TEKS2, wraplength=920, justify="left"
                 ).pack(anchor="w", padx=16, pady=(0, 10))
        self._kartu_rumus(parent,
            "T(x,y) = [ jumlah( w(i,j) x I_blur(i,j) ) / jumlah( w(i,j) ) ] - C",
            ["w(i,j) = bobot Gaussian pada jendela di sekitar (x,y)",
             "I_blur = citra setelah median blur 3x3;  C = konstanta pengurang"])
        self._kartu_rumus(parent, "I_biner(x,y) = 255 jika I(x,y) > T(x,y);  jika tidak, 0")

        r = viz_titik_agt(self.gray, titik=self.titik_agt)

        body = tk.Frame(parent, bg=BG)
        body.pack(fill="x", padx=16)
        body.columnconfigure(0, weight=3)
        body.columnconfigure(1, weight=2)

        kiri = tk.Frame(body, bg=PANEL)
        kiri.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        tk.Label(kiri, text="Klik gambar untuk memindahkan titik yang dihitung:",
                 font=F_KECIL, bg=PANEL, fg=TEKS2).pack(anchor="w", padx=12, pady=(12, 4))
        self._preview_klik(kiri, self.gray, self._klik_titik_agt, lebar_maks=380, tinggi_maks=340,
                            titik=r["titik"], kotak_ukuran=AGT_BLOCK_SIZE)

        kanan = tk.Frame(body, bg=PANEL)
        kanan.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        tk.Label(kanan, text=f"Jendela {AGT_BLOCK_SIZE}x{AGT_BLOCK_SIZE} piksel di sekitar titik",
                 font=F_KECIL, bg=PANEL, fg=TEKS2).pack(anchor="w", padx=12, pady=(12, 6))
        pasangan = tk.Frame(kanan, bg=PANEL)
        pasangan.pack(padx=12, pady=(0, 16))
        self._thumb(pasangan, r["window_asli_crop"], "Jendela asli", w=140, h=140).pack(side="left", padx=(0, 8))
        self._thumb(pasangan, r["window_blur_crop"], "Setelah median blur 3x3", w=140, h=140).pack(side="left")

        txt = _buat_terminal_panel(parent, warna_aksen=AKSEN, tinggi=14,
            judul=f"Perhitungan pada Titik (x={r['titik'][1]}, y={r['titik'][0]})")
        self._isi_terminal_agt(txt, r)
        self._garis(parent)

        self._judul_sub(parent, "Hasil Akhir (Seluruh Citra)")
        hasil_row = tk.Frame(parent, bg=BG)
        hasil_row.pack(padx=16, pady=(0, 20), anchor="w")
        self._thumb(hasil_row, self.gray, "Sebelum (grayscale)", w=160, h=160).pack(side="left", padx=(0, 10))
        self._thumb(hasil_row, self._binary, "Sesudah (AGT)",
                    w=160, h=160).pack(side="left")

    # ---------------------------------------------------- panel: Line Removal
    def _panel_line(self, parent):
        judul_seksi(parent, "4. Line Removal — Penghapusan Garis Tabel", warna_aksen=AKSEN2)
        self._kartu_rumus(parent, "Rg = P_garis / P_total",
            ["P_garis = jumlah piksel pada mask garis;  P_total = jumlah seluruh piksel citra",
             f"Line removal hanya dijalankan bila Rg > {LINE_RATIO_THRESHOLD}"])

        r = self._line
        txt = _buat_terminal_panel(parent, warna_aksen=AKSEN2, tinggi=11,
                                    judul="Perhitungan Rasio Garis (Keputusan Aktivasi)")
        self._isi_terminal_line(txt, r)
        badge_row = tk.Frame(parent, bg=BG)
        badge_row.pack(anchor="w", padx=16, pady=(10, 8))
        self._badge_keputusan(badge_row, r["aktif"],
            "LINE REMOVAL DIJALANKAN" if r["aktif"] else "LINE REMOVAL DILEWATI").pack(side="left")
        self._garis(parent)

        self._judul_sub(parent, "Tahapan Proses")
        tahapan = [
            (r["biner"],       "1. Citra Biner",          "hasil AGT"),
            (r["inv"],         "2. Inversi",              "I' = 255 - I_biner"),
            (r["buka_h"],      "3. Opening Horizontal",   f"kernel {r['kh_size'][0]}x{r['kh_size'][1]}"),
            (r["buka_v"],      "4. Opening Vertikal",     f"kernel {r['kv_size'][0]}x{r['kv_size'][1]}"),
            (r["mask"],        "5. Mask Gabungan",        "M = M_h + M_v"),
            (r["mask_dilasi"], "6. Mask Setelah Dilasi",  "M' = Dilasi(M, 3x3)"),
            (r["final"],       "7. Hasil Akhir",          "masking + invers kembali"),
        ]
        filmstrip = tk.Frame(parent, bg=BG)
        filmstrip.pack(padx=16, pady=(0, 16), anchor="w")
        for idx, (arr, judul, ket) in enumerate(tahapan):
            col, row = idx % 4, idx // 4
            card = tk.Frame(filmstrip, bg=PANEL)
            card.grid(row=row, column=col, padx=6, pady=6, sticky="n")
            tinggi_gbr = max(1, int(150 * arr.shape[0] / arr.shape[1]))
            foto = self._foto(arr, w=150, h=tinggi_gbr)
            tk.Label(card, image=foto, bg=PANEL).pack(padx=6, pady=(6, 2))
            tk.Label(card, text=judul, font=("Segoe UI", 8, "bold"), bg=PANEL, fg=PUTIH).pack()
            tk.Label(card, text=ket, font=F_KECIL, bg=PANEL, fg=TEKS2).pack(pady=(0, 6))

        if not r["aktif"]:
            tk.Label(parent, text="Rg berada di bawah threshold, sehingga tahap 5-7 di atas tetap "
                                  "dihitung untuk ilustrasi, namun HASIL AKHIR pipeline tetap memakai "
                                  "citra biner apa adanya karena keputusannya DILEWATI.",
                     font=F_KECIL, bg=BG, fg=KUNING, wraplength=920, justify="left"
                     ).pack(anchor="w", padx=16, pady=(0, 16))
        else:
            tk.Frame(parent, bg=BG, height=16).pack()

    # ---------------------------------------------------- panel: CER & WER
    def _panel_cerwer(self, parent):
        judul_seksi(parent, "5. CER & WER — Evaluasi Akurasi OCR", warna_aksen=AKSEN)
        self._kartu_rumus(parent, "CER = (Sc + Dc + Ic) / Nc",
            ["Sc = jumlah karakter tersubstitusi,  Dc = jumlah karakter terhapus,  "
             "Ic = jumlah karakter tertambah (dibanding referensi)",
             "Nc = jumlah total karakter pada teks referensi (ground truth)"])
        self._kartu_rumus(parent, "WER = (Sw + Dw + Iw) / Nw",
            ["Sw, Dw, Iw = jumlah kata tersubstitusi, terhapus, tertambah",
             "Nw = jumlah total kata pada teks referensi (ground truth)"])
        self._kartu_rumus(parent,
            "d(i,j) = maks(i,j) jika salah satu teks kosong;  jika tidak,  "
            "d(i,j) = min[ d(i-1,j)+1,  d(i,j-1)+1,  d(i-1,j-1)+cost ]",
            ["cost = 0 jika karakter/kata sama, 1 jika berbeda — ini algoritma Levenshtein Distance",
             "d(i,j) ditelusuri balik (backtrack) untuk memecah totalnya menjadi Sc/Dc/Ic atau Sw/Dw/Iw"])

        self._judul_sub(parent, "Masukkan Teks untuk Dihitung")
        tk.Label(parent, text="Bisa diisi manual (contoh bawaan ini sama seperti latihan Excel-mu), atau klik "
                              "\"Jalankan OCR\" untuk mengambil hasil OCR sungguhan dari gambar yang sedang dimuat.",
                 font=F_KECIL, bg=BG, fg=TEKS2, wraplength=920, justify="left"
                 ).pack(anchor="w", padx=16, pady=(0, 10))

        body = tk.Frame(parent, bg=BG)
        body.pack(fill="x", padx=16)
        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, weight=1)

        kiri = tk.Frame(body, bg=PANEL)
        kiri.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        tk.Label(kiri, text="Ground Truth (Referensi)", font=F_JUDUL, bg=PANEL, fg=PUTIH
                 ).pack(anchor="w", padx=12, pady=(10, 4))
        gt_wrap = tk.Frame(kiri, bg=CARD)
        gt_wrap.pack(fill="x", padx=12, pady=(0, 8))
        self._txt_gt_cerwer = tk.Text(gt_wrap, font=F_MONO, height=3, bg=CARD, fg=TEKS,
                                       insertbackground=TEKS, relief="flat", bd=0, wrap="word",
                                       padx=8, pady=8)
        gt_sb = ttk.Scrollbar(gt_wrap, orient="vertical", command=self._txt_gt_cerwer.yview)
        self._txt_gt_cerwer.configure(yscrollcommand=gt_sb.set)
        self._txt_gt_cerwer.pack(side="left", fill="both", expand=True)
        gt_sb.pack(side="right", fill="y")
        self._txt_gt_cerwer.insert("1.0", self.teks_gt_input)
        buat_tombol(kiri, "📁 Upload Ground Truth (.txt)", self._klik_upload_gt_cerwer,
                    warna="#21262d", lebar=27).pack(anchor="w", padx=12, pady=(0, 12))

        kanan = tk.Frame(body, bg=PANEL)
        kanan.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        tk.Label(kanan, text="Hasil OCR (Hipotesis)", font=F_JUDUL, bg=PANEL, fg=PUTIH
                 ).pack(anchor="w", padx=12, pady=(10, 4))
        hyp_wrap = tk.Frame(kanan, bg=CARD)
        hyp_wrap.pack(fill="x", padx=12, pady=(0, 8))
        self._txt_hyp_cerwer = tk.Text(hyp_wrap, font=F_MONO, height=3, bg=CARD, fg=TEKS,
                                        insertbackground=TEKS, relief="flat", bd=0, wrap="word",
                                        padx=8, pady=8)
        hyp_sb = ttk.Scrollbar(hyp_wrap, orient="vertical", command=self._txt_hyp_cerwer.yview)
        self._txt_hyp_cerwer.configure(yscrollcommand=hyp_sb.set)
        self._txt_hyp_cerwer.pack(side="left", fill="both", expand=True)
        hyp_sb.pack(side="right", fill="y")
        self._txt_hyp_cerwer.insert("1.0", self.teks_hyp_input)
        buat_tombol(kanan, "🔤 Jalankan OCR pada Gambar Ini", self._klik_ocr_gambar_cerwer,
                    warna="#21262d", lebar=27).pack(anchor="w", padx=12, pady=(0, 12))

        tengah = tk.Frame(parent, bg=BG)
        tengah.pack(pady=(12, 20))
        buat_tombol(tengah, "▶  Hitung CER & WER", self._klik_hitung_cerwer,
                    warna=AKSEN, lebar=22).pack()

        self._garis(parent)

        hasil = self._cerwer_hasil
        if hasil is None:
            return

        self._judul_sub(parent, "Perhitungan Level Karakter")
        txt1 = _buat_terminal_panel(parent, warna_aksen=AKSEN, tinggi=10,
                                     judul="Rincian CER")
        self._isi_terminal_cer(txt1, hasil)

        if len(hasil["gt"]) <= 18 and len(hasil["hyp"]) <= 18 and hasil["nc"] > 0:
            tk.Label(parent, text="Matriks Levenshtein Level Karakter "
                                  "(kotak bertepi = jalur perhitungan optimal):",
                     font=F_KECIL, bg=BG, fg=TEKS2).pack(anchor="w", padx=16, pady=(10, 6))
            self._grid_matriks_levenshtein(parent, hasil["char"]).pack(anchor="w", padx=16, pady=(0, 6))
            self._legenda_matriks(parent)
        else:
            tk.Label(parent, text="(Matriks tidak ditampilkan karena teks referensi/hipotesis melebihi 18 "
                                  "karakter — akan terlalu besar untuk ditampilkan dengan jelas. Angka CER "
                                  "di atas tetap dihitung dari keseluruhan teks.)",
                     font=F_KECIL, bg=BG, fg=TEKS2, wraplength=920, justify="left"
                     ).pack(anchor="w", padx=16, pady=(10, 16))

        self._garis(parent)

        self._judul_sub(parent, "Perhitungan Level Kata")
        txt2 = _buat_terminal_panel(parent, warna_aksen=AKSEN2, tinggi=10,
                                     judul="Rincian WER")
        self._isi_terminal_wer(txt2, hasil)

        if len(hasil["gt"].split()) <= 12 and len(hasil["hyp"].split()) <= 12 and hasil["nw"] > 0:
            tk.Label(parent, text="Matriks Levenshtein Level Kata:",
                     font=F_KECIL, bg=BG, fg=TEKS2).pack(anchor="w", padx=16, pady=(10, 6))
            self._grid_matriks_levenshtein(parent, hasil["kata"]).pack(anchor="w", padx=16, pady=(0, 6))
            self._legenda_matriks(parent)
        else:
            tk.Label(parent, text="(Matriks kata tidak ditampilkan karena jumlah kata melebihi 12.)",
                     font=F_KECIL, bg=BG, fg=TEKS2, wraplength=920, justify="left"
                     ).pack(anchor="w", padx=16, pady=(10, 16))

        self._garis(parent)

    def _grid_matriks_levenshtein(self, parent, hasil):
        ref, hyp, m, n, d = hasil["ref"], hasil["hyp"], hasil["m"], hasil["n"], hasil["matriks"]
        jalur, jenis_map = hasil["jalur"], hasil["jalur_jenis"]
        bg_jenis = {"cocok": "#132b1c", "substitusi": "#332a0f", "hapus": "#331313", "tambah": "#161b3a"}
        fg_jenis = {"cocok": HIJAU, "substitusi": KUNING, "hapus": MERAH, "tambah": AKSEN2}

        def tampil(tok):
            return "·" if tok == " " else str(tok)

        grid = tk.Frame(parent, bg=BG)
        tk.Label(grid, text="", width=4, bg=BG).grid(row=0, column=0)
        tk.Label(grid, text="ε", width=3, font=F_MONO, bg=CARD, fg=TEKS2).grid(
            row=0, column=1, padx=1, pady=1)
        for j, tok in enumerate(hyp, start=2):
            tk.Label(grid, text=tampil(tok), width=3, font=F_MONO, bg=CARD, fg=PUTIH
                     ).grid(row=0, column=j, padx=1, pady=1)

        for i in range(m + 1):
            judul_baris = "ε" if i == 0 else tampil(ref[i - 1])
            tk.Label(grid, text=judul_baris, width=4, font=F_MONO, bg=CARD,
                     fg=(TEKS2 if i == 0 else PUTIH)).grid(row=i + 1, column=0, padx=1, pady=1)
            for j in range(n + 1):
                aktif = (i, j) in jalur
                jenis = jenis_map.get((i, j))
                bgc = bg_jenis.get(jenis, PANEL) if aktif else PANEL
                fgc = fg_jenis.get(jenis, TEKS) if aktif else TEKS3
                tk.Label(grid, text=str(d[i][j]), width=3, font=F_MONO, bg=bgc, fg=fgc,
                         relief="solid" if aktif else "flat", bd=1 if aktif else 0
                         ).grid(row=i + 1, column=j + 1, padx=1, pady=1)
        return grid

    def _legenda_matriks(self, parent):
        f = tk.Frame(parent, bg=BG)
        f.pack(anchor="w", padx=16, pady=(6, 16))
        for warna, teks in [(HIJAU, "cocok"), (KUNING, "substitusi"), (MERAH, "hapus"), (AKSEN2, "tambah")]:
            tk.Frame(f, bg=warna, width=12, height=12).pack(side="left", padx=(0, 4))
            tk.Label(f, text=teks, font=F_KECIL, bg=BG, fg=TEKS2).pack(side="left", padx=(0, 12))

    def _isi_terminal_cer(self, txt, hasil):
        r = hasil["char"]
        self._tulis(txt, f"Ground Truth (dinormalisasi) : \"{hasil['gt']}\"\n", "abu")
        self._tulis(txt, f"Hasil OCR    (dinormalisasi) : \"{hasil['hyp']}\"\n\n", "abu")
        if hasil["nc"] == 0:
            self._tulis(txt, "  Ground Truth kosong -> CER tidak dapat dihitung\n", "kuning")
            return
        self._tulis(txt, f"  Sc (substitusi)     = {r['sub']}\n", "normal")
        self._tulis(txt, f"  Dc (hapus)          = {r['hapus']}\n", "normal")
        self._tulis(txt, f"  Ic (tambah)         = {r['tambah']}\n", "normal")
        self._tulis(txt, f"  Nc (panjang GT)     = {hasil['nc']}\n\n", "normal")
        self._tulis(txt, f"  CER = ({r['sub']} + {r['hapus']} + {r['tambah']}) / {hasil['nc']} = ", "normal")
        self._tulis(txt, f"{hasil['cer']}%\n", "aksen")

    def _isi_terminal_wer(self, txt, hasil):
        r = hasil["kata"]
        self._tulis(txt, f"Kata GT  : {hasil['gt'].split()}\n", "abu")
        self._tulis(txt, f"Kata OCR : {hasil['hyp'].split()}\n\n", "abu")
        if hasil["nw"] == 0:
            self._tulis(txt, "  Ground Truth kosong -> WER tidak dapat dihitung\n", "kuning")
            return
        self._tulis(txt, f"  Sw (substitusi)     = {r['sub']}\n", "normal")
        self._tulis(txt, f"  Dw (hapus)          = {r['hapus']}\n", "normal")
        self._tulis(txt, f"  Iw (tambah)         = {r['tambah']}\n", "normal")
        self._tulis(txt, f"  Nw (jumlah kata GT) = {hasil['nw']}\n\n", "normal")
        self._tulis(txt, f"  WER = ({r['sub']} + {r['hapus']} + {r['tambah']}) / {hasil['nw']} = ", "normal")
        self._tulis(txt, f"{hasil['wer']}%\n", "aksen2")

    # ---------------------------------------------------- callback CER & WER
    def _klik_hitung_cerwer(self):
        self.teks_gt_input  = self._txt_gt_cerwer.get("1.0", "end").rstrip("\n")
        self.teks_hyp_input = self._txt_hyp_cerwer.get("1.0", "end").rstrip("\n")
        if not self.teks_gt_input.strip():
            messagebox.showinfo("Info", "Isi teks Ground Truth terlebih dahulu.")
            return
        self._hitung_cerwer()
        self._tampilkan_langkah("cerwer")

    def _klik_ocr_gambar_cerwer(self):
        if self.img_asli is None:
            messagebox.showinfo("Info", "Pilih gambar terlebih dahulu.")
            return
        self.teks_gt_input = self._txt_gt_cerwer.get("1.0", "end").rstrip("\n")
        self.winfo_toplevel().configure(cursor="watch")
        self.update()
        try:
            self.teks_hyp_input = run_ocr(self.img_asli)
        except Exception as exc:
            self.winfo_toplevel().configure(cursor="")
            messagebox.showerror("Error", f"Gagal menjalankan OCR:\n{exc}")
            return
        self.winfo_toplevel().configure(cursor="")
        self._hitung_cerwer()
        self._tampilkan_langkah("cerwer")

    def _klik_upload_gt_cerwer(self):
        path = filedialog.askopenfilename(
            filetypes=[("Teks", "*.txt"), ("Semua file", "*.*")])
        if not path:
            return
        self.teks_hyp_input = self._txt_hyp_cerwer.get("1.0", "end").rstrip("\n")
        try:
            with open(path, encoding="utf-8") as f:
                self.teks_gt_input = f.read()
        except Exception:
            with open(path, encoding="latin-1") as f:
                self.teks_gt_input = f.read()
        self._hitung_cerwer()
        self._tampilkan_langkah("cerwer")


class MainApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("PRAOCR")
        self.configure(bg=NAV_BG)
        self.state("zoomed")

        self._halaman_aktif = None
        self._bangun_navbar()
        self._bangun_halaman()
        self._tampil("prepro")

    def _bangun_navbar(self):
        nav = tk.Frame(self, bg=NAV_BG, height=44)
        nav.pack(fill="x")
        nav.pack_propagate(False)

        tk.Frame(self, bg=BORDER, height=1).pack(fill="x")

        tabs = [
            ("prepro", "🖼   Preprocessing & OCR", AKSEN),
            ("visual", "🧮   Visualisasi Proses",  TEAL),
            ("eval",   "📊   Evaluasi",             AKSEN2),
            ("raster", "📄   Rasterisasi PDF",      KUNING),
        ]
        self._tab_data = {}
        for key, label, warna in tabs:
            btn, ind = self._buat_tab(nav, label, key, warna)
            self._tab_data[key] = (btn, ind, warna)

    def _buat_tab(self, parent, text, key, warna_aktif):
        wrap = tk.Frame(parent, bg=NAV_BG)
        wrap.pack(side="left")
        btn = tk.Button(
            wrap, text=text, font=F_LABEL,
            bg=NAV_BG, fg=TEKS2,
            activebackground=PANEL, activeforeground=PUTIH,
            relief="flat", bd=0, padx=18, pady=10,
            cursor="hand2",
            command=lambda: self._tampil(key))
        btn.pack()
        ind = tk.Frame(wrap, bg=NAV_BG, height=2)
        ind.pack(fill="x")
        return btn, ind

    def _bangun_halaman(self):
        self.frame_prepro = AppFrame(self)
        self.frame_visual = VisualisasiFrame(self)
        self.frame_eval   = EvaluasiFrame(self)
        self.frame_raster = RasterisasiFrame(self)

    def _tampil(self, halaman: str):
        if halaman == self._halaman_aktif:
            return
        self._halaman_aktif = halaman

        for key, (btn, ind, warna) in self._tab_data.items():
            aktif = (key == halaman)
            btn.configure(fg=PUTIH if aktif else TEKS2,
                          bg=PANEL  if aktif else NAV_BG)
            ind.configure(bg=warna if aktif else NAV_BG)

        for frame in [self.frame_prepro, self.frame_visual, self.frame_eval, self.frame_raster]:
            frame.pack_forget()

        {
            "prepro": self.frame_prepro,
            "visual": self.frame_visual,
            "eval":   self.frame_eval,
            "raster": self.frame_raster,
        }[halaman].pack(fill="both", expand=True)


if __name__ == "__main__":
    app = MainApp()
    app.mainloop()