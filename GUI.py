import os
import re
import cv2
import numpy as np
import pytesseract
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
CONFIG_TESSERACT = "--oem 3 --psm 6 -l ind+eng"

# ── Palet warna ────────────────────────────────────────────────────────────────
BG          = "#0d1117"
PANEL       = "#161b22"
CARD        = "#1c2130"
BORDER      = "#30363d"
AKSEN       = "#4f87f7"
AKSEN2      = "#7c5aed"
HIJAU       = "#2ea043"
MERAH       = "#da3633"
KUNING      = "#d29922"
TEKS        = "#e6edf3"
TEKS2       = "#7d8590"
TEKS3       = "#444c56"
PUTIH       = "#ffffff"

F_JUDUL     = ("Segoe UI", 10, "bold")
F_LABEL     = ("Segoe UI", 9)
F_KECIL     = ("Segoe UI", 8)
F_MONO      = ("Consolas", 9)
F_BESAR     = ("Segoe UI", 22, "bold")
F_BADGE     = ("Segoe UI", 7, "bold")


# ══════════════════════════════════════════════════════════════════════════════
# PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def to_grayscale(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return image.copy()


def clahe(gray: np.ndarray) -> tuple[np.ndarray, bool]:
    if float(np.std(gray)) >= 30:
        return gray, False
    enh = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(4, 4))
    return enh.apply(gray), True


def binarize(gray: np.ndarray) -> np.ndarray:
    gray = cv2.medianBlur(gray, 3)
    return cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 111, 55
    )


def remove_lines(binary: np.ndarray) -> tuple[np.ndarray, bool]:
    h, w = binary.shape
    inv  = ~binary
    kh   = cv2.getStructuringElement(cv2.MORPH_RECT, (max(10, w // 40), 1))
    kv   = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(10, h // 40)))
    mask = cv2.add(
        cv2.morphologyEx(inv, cv2.MORPH_OPEN, kh),
        cv2.morphologyEx(inv, cv2.MORPH_OPEN, kv)
    )
    if np.count_nonzero(mask) / (h * w) <= 0.003:
        return binary, False
    mask  = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
    hasil = cv2.bitwise_and(inv, inv, mask=cv2.bitwise_not(mask))
    return ~hasil, True


def pipeline_adaptive(image: np.ndarray) -> tuple[np.ndarray, dict]:
    """Jalankan pipeline penuh, kembalikan (citra, log aktif/tidak)."""
    log = {}
    img = to_grayscale(image);          log["grayscale"]    = True
    img, log["clahe"]        = clahe(img)
    img = binarize(img);                log["binarize"]     = True
    img, log["line_removal"] = remove_lines(img)
    return img, log


def pipeline_dengan_override(
    image: np.ndarray,
    log_asli: dict,
    dimatikan: set,
) -> np.ndarray:
    """
    Jalankan ulang pipeline dengan beberapa teknik dimatikan secara paksa.

    Aturan:
    - Teknik yang di log_asli=False (tidak aktif karena threshold) → selalu skip,
      tidak peduli isi dimatikan.
    - Teknik yang di log_asli=True (aktif) tapi ada di dimatikan → skip.
    - Teknik yang di log_asli=True dan tidak ada di dimatikan → jalankan normal.
    - grayscale dan binarize selalu dijalankan (wajib).
    """
    img = to_grayscale(image)

    # clahe
    if log_asli.get("clahe") and "clahe" not in dimatikan:
        img, _ = clahe(img)

    # binarize — selalu aktif
    img = binarize(img)

    # line_removal
    if log_asli.get("line_removal") and "line_removal" not in dimatikan:
        img, _ = remove_lines(img)

    return img


def clahe_paksa(gray: np.ndarray) -> np.ndarray:
    """Jalankan CLAHE tanpa cek threshold — untuk paksa aktif."""
    enh = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(4, 4))
    return enh.apply(gray)


def remove_lines_paksa(binary: np.ndarray) -> np.ndarray:
    """Jalankan remove_lines tanpa cek threshold — untuk paksa aktif."""
    h, w = binary.shape
    inv  = ~binary
    kh   = cv2.getStructuringElement(cv2.MORPH_RECT, (max(10, w // 40), 1))
    kv   = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(10, h // 40)))
    mask = cv2.add(
        cv2.morphologyEx(inv, cv2.MORPH_OPEN, kh),
        cv2.morphologyEx(inv, cv2.MORPH_OPEN, kv)
    )
    mask  = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
    hasil = cv2.bitwise_and(inv, inv, mask=cv2.bitwise_not(mask))
    return ~hasil


def pipeline_manual(image: np.ndarray, aktif: dict) -> tuple:
    """
    Jalankan pipeline dengan kontrol penuh tiap teknik.
    aktif: dict {nama: True/False} menentukan teknik mana yang dijalankan.

    Kembalikan (img_tampil, img_ocr):
      - img_tampil : ditampilkan di canvas
      - img_ocr    : dipakai untuk OCR

    Jika grayscale nonaktif → img_tampil dan img_ocr adalah RGB asli
    (tidak ada preprocessing sama sekali, OCR pada gambar RGB mentah).
    """
    if not aktif.get("grayscale", True):
        # Grayscale nonaktif → tidak ada preprocessing, OCR pada RGB asli
        img_rgb = image.copy()
        return img_rgb, img_rgb

    # Grayscale aktif → mulai pipeline
    img = to_grayscale(image)

    if aktif.get("clahe", False):
        img = clahe_paksa(img)

    if aktif.get("binarize", True):
        img = binarize(img)

    if aktif.get("line_removal", False):
        img = remove_lines_paksa(img)

    return img, img


# ══════════════════════════════════════════════════════════════════════════════
# OCR & METRIK
# ══════════════════════════════════════════════════════════════════════════════

def run_ocr(image: np.ndarray) -> str:
    pil = Image.fromarray(image)
    t   = pytesseract.image_to_string(pil, config=CONFIG_TESSERACT)
    t   = t.replace("|", "")
    t   = re.sub(r'\n\s*\n+', '\n', t)
    return "\n".join(line.strip() for line in t.splitlines()).strip()


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)  # newline & spasi → spasi tunggal
    return text.strip()


def hitung_cer_wer(hypothesis: str, reference: str) -> dict:
    if not reference.strip():
        return {"cer": None, "wer": None}
    try:
        from jiwer import cer, wer
        h = normalize_text(hypothesis)
        r = normalize_text(reference)
        return {
            "cer": round(cer(r, h) * 100, 2),
            "wer": round(wer(r, h) * 100, 2),
        }
    except Exception:
        return {"cer": None, "wer": None}


def analisis_gambar(img_gray: np.ndarray) -> dict:
    """Hitung metrik analisis gambar dari citra grayscale."""
    # Intensitas cahaya: rata-rata kecerahan piksel (0-255)
    intensitas = round(float(np.mean(img_gray)), 2)

    # Tingkat noise: std dari residual Gaussian blur
    blurred = cv2.GaussianBlur(img_gray, (5, 5), 0)
    noise = round(float(np.std(
        img_gray.astype(np.float32) - blurred.astype(np.float32)
    )), 2)

    # Deteksi garis: rasio piksel garis terhadap total piksel
    if len(img_gray.shape) == 3:
        img_gray = cv2.cvtColor(img_gray, cv2.COLOR_RGB2GRAY)

    if len(img_gray.shape) != 2:
        img_gray = to_grayscale(img_gray)

    if img_gray.dtype != np.uint8:
        img_gray_u8 = np.clip(img_gray, 0, 255).astype(np.uint8)
    else:
        img_gray_u8 = img_gray

    if img_gray_u8.dtype != np.uint8 or len(img_gray_u8.shape) != 2:
        img_gray_u8 = np.clip(img_gray_u8, 0, 255)
        if len(img_gray_u8.shape) == 3:
            img_gray_u8 = cv2.cvtColor(img_gray_u8, cv2.COLOR_RGB2GRAY)
        img_gray_u8 = img_gray_u8.astype(np.uint8)

    temp = cv2.medianBlur(img_gray_u8, 3)
    if temp.dtype != np.uint8 or len(temp.shape) != 2:
        temp = np.clip(temp, 0, 255).astype(np.uint8)
        if len(temp.shape) == 3:
            temp = cv2.cvtColor(temp, cv2.COLOR_RGB2GRAY)

    if temp.dtype != np.uint8 or len(temp.shape) != 2:
        temp = np.clip(temp, 0, 255)
        if len(temp.shape) == 3:
            temp = cv2.cvtColor(temp.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        temp = temp.astype(np.uint8)

    temp_bin = cv2.adaptiveThreshold(
        temp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 111, 55
    )

    h, w  = temp_bin.shape
    inv   = cv2.bitwise_not(temp_bin)
    kh    = cv2.getStructuringElement(cv2.MORPH_RECT, (max(10, w // 40), 1))
    kv    = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(10, h // 40)))
    mask  = cv2.add(
        cv2.morphologyEx(inv, cv2.MORPH_OPEN, kh),
        cv2.morphologyEx(inv, cv2.MORPH_OPEN, kv),
    )
    rasio_garis = round(np.count_nonzero(mask) / (h * w) * 100, 3)
    terdeteksi  = rasio_garis > 0.3

    return {
        "intensitas": intensitas,
        "noise": noise,
        "rasio_garis": rasio_garis,
        "ada_garis": terdeteksi,
    }


def pil_ke_tk(pil_img: Image.Image, max_w: int, max_h: int) -> ImageTk.PhotoImage:
    w, h  = pil_img.size
    skala = min(max_w / w, max_h / h, 1.0)
    if skala < 1.0:
        pil_img = pil_img.resize((int(w * skala), int(h * skala)), Image.LANCZOS)
    return ImageTk.PhotoImage(pil_img)


# ══════════════════════════════════════════════════════════════════════════════
# ZOOMABLE CANVAS
# ══════════════════════════════════════════════════════════════════════════════

class ZoomableCanvas(tk.Canvas):
    """Canvas dengan fitur zoom dan pan."""

    ZOOM_STEP   = 1.25
    ZOOM_MIN    = 0.1
    ZOOM_MAX    = 20.0

    def __init__(self, parent, **kw):
        kw.setdefault("bg", "#0a0d14")
        kw.setdefault("highlightthickness", 0)
        kw.setdefault("cursor", "crosshair")
        super().__init__(parent, **kw)

        self._pil_src  : Image.Image | None = None
        self._tk_img   : ImageTk.PhotoImage | None = None
        self._zoom     : float = 1.0
        self._fit_scale: float = 1.0
        self._offset_x : float = 0.0
        self._offset_y : float = 0.0
        self._drag_start: tuple[int, int] | None = None

        self.bind("<Configure>",       self._on_resize)
        self.bind("<MouseWheel>",      self._on_scroll_win)
        self.bind("<Button-4>",        self._on_scroll_up)
        self.bind("<Button-5>",        self._on_scroll_down)
        self.bind("<ButtonPress-1>",   self._on_drag_start)
        self.bind("<B1-Motion>",       self._on_drag_move)
        self.bind("<ButtonRelease-1>", self._on_drag_end)

    def set_image(self, img: np.ndarray, gray: bool = False):
        if gray and len(img.shape) == 2:
            self._pil_src = Image.fromarray(img).convert("RGB")
        else:
            self._pil_src = Image.fromarray(img)
        self._reset_zoom()

    def clear(self):
        self._pil_src = None
        self._tk_img  = None
        self.delete("all")

    def reset_zoom(self):
        self._reset_zoom()

    def _reset_zoom(self):
        self.update_idletasks()
        cw = self.winfo_width()  or 500
        ch = self.winfo_height() or 400

        if self._pil_src is None:
            return

        iw, ih = self._pil_src.size
        self._fit_scale = min(cw / iw, ch / ih, 1.0)
        self._zoom      = 1.0
        self._offset_x  = cw / 2
        self._offset_y  = ch / 2
        self._render()

    def _render(self):
        if self._pil_src is None:
            return

        self.update_idletasks()
        cw = self.winfo_width()  or 500
        ch = self.winfo_height() or 400

        scale  = self._fit_scale * self._zoom
        iw, ih = self._pil_src.size
        nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))

        resized = self._pil_src.resize((nw, nh), Image.LANCZOS)
        self._tk_img = ImageTk.PhotoImage(resized)

        self.delete("img")
        self.create_image(
            self._offset_x, self._offset_y,
            image=self._tk_img, anchor="center", tags="img"
        )

    def _on_resize(self, event):
        if self._pil_src is not None:
            iw, ih = self._pil_src.size
            self._fit_scale = min(event.width / iw, event.height / ih, 1.0)
            self._offset_x = event.width  / 2
            self._offset_y = event.height / 2
            self._render()

    def _zoom_at(self, x: int, y: int, faktor: float):
        new_zoom = max(self.ZOOM_MIN, min(self.ZOOM_MAX, self._zoom * faktor))
        ratio    = new_zoom / self._zoom
        self._offset_x = x + (self._offset_x - x) * ratio
        self._offset_y = y + (self._offset_y - y) * ratio
        self._zoom     = new_zoom
        self._render()

    def _on_scroll_win(self, event):
        faktor = self.ZOOM_STEP if event.delta > 0 else 1 / self.ZOOM_STEP
        self._zoom_at(event.x, event.y, faktor)

    def _on_scroll_up(self, event):
        self._zoom_at(event.x, event.y, self.ZOOM_STEP)

    def _on_scroll_down(self, event):
        self._zoom_at(event.x, event.y, 1 / self.ZOOM_STEP)

    def _on_drag_start(self, event):
        self._drag_start = (event.x, event.y)
        self.configure(cursor="fleur")

    def _on_drag_move(self, event):
        if self._drag_start is None:
            return
        dx = event.x - self._drag_start[0]
        dy = event.y - self._drag_start[1]
        self._offset_x   += dx
        self._offset_y   += dy
        self._drag_start  = (event.x, event.y)
        self._render()

    def _on_drag_end(self, event):
        self._drag_start = None
        self.configure(cursor="crosshair")


# ══════════════════════════════════════════════════════════════════════════════
# WIDGET HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def buat_tombol(parent, text, cmd, warna=AKSEN, lebar=14):
    b = tk.Button(
        parent, text=text, command=cmd,
        font=F_LABEL, bg=warna, fg=PUTIH,
        activebackground=warna, activeforeground=PUTIH,
        relief="flat", bd=0, width=lebar,
        cursor="hand2", padx=8, pady=5,
    )
    def darken(c):
        c = c.lstrip("#")
        r, g, b = int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)
        return f"#{max(0,r-25):02x}{max(0,g-25):02x}{max(0,b-25):02x}"
    b.bind("<Enter>", lambda e: b.config(bg=darken(warna)))
    b.bind("<Leave>", lambda e: b.config(bg=warna))
    return b


def judul_seksi(parent, teks, warna_aksen=AKSEN, canvas_zoom=None):
    baris = tk.Frame(parent, bg=PANEL)
    baris.pack(fill="x", padx=12, pady=(10, 6))
    tk.Frame(baris, bg=warna_aksen, width=3, height=16).pack(side="left", padx=(0, 8))
    tk.Label(baris, text=teks, font=F_JUDUL, bg=PANEL, fg=PUTIH).pack(side="left")
    if canvas_zoom is not None:
        tk.Button(
            baris, text="⊡ Reset Zoom",
            command=canvas_zoom.reset_zoom,
            font=F_KECIL, bg="#21262d", fg=TEKS2,
            activebackground="#30363d", activeforeground=TEKS,
            relief="flat", bd=0, padx=6, pady=2, cursor="hand2",
        ).pack(side="right")


# ══════════════════════════════════════════════════════════════════════════════
# APLIKASI UTAMA
# ══════════════════════════════════════════════════════════════════════════════

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Evaluasi Preprocessing OCR — Tugas Akhir")
        self.configure(bg=BG)
        self.state("zoomed")

        # State
        self.img_asli      = None   # np.ndarray RGB
        self.img_prep      = None   # np.ndarray grayscale
        self.teks_gt       = ""
        self.teks_ocr_asli = ""
        self.teks_ocr_prep = ""
        self.log_teknik    = {}
        self.dimatikan     = set()

        self.aktif_teknik  = {}
        self.aktif_awal    = {}

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
        tk.Label(kiri, text="Evaluasi Preprocessing Adaptif OCR",
                 font=("Segoe UI", 13, "bold"), bg=PANEL, fg=PUTIH).pack(side="left")
        tk.Label(kiri, text="  —  Alat Bantu Penulisan Laporan Tugas Akhir",
                 font=F_LABEL, bg=PANEL, fg=TEKS2).pack(side="left")

        kanan = tk.Frame(hdr, bg=PANEL)
        kanan.pack(side="right", padx=16)

        buat_tombol(kanan, "📁  Pilih Gambar", self._pilih_gambar,
                    warna="#21262d", lebar=16).pack(side="left", padx=4)
        buat_tombol(kanan, "📄  Upload Ground Truth", self._pilih_gt,
                    warna="#21262d", lebar=22).pack(side="left", padx=4)
        buat_tombol(kanan, "▶  Jalankan", self._jalankan,
                    warna=AKSEN, lebar=14).pack(side="left", padx=4)
        buat_tombol(kanan, "↺  Reset", self._reset,
                    warna="#21262d", lebar=10).pack(side="left", padx=4)

    def _bangun_panel_gambar_asli(self, parent):
        f = tk.Frame(parent, bg=PANEL)
        f.grid(row=0, column=0, sticky="nsew", padx=(0, 4), pady=4)
        f.grid_propagate(False)

        self.canvas_asli = ZoomableCanvas(f)

        hdr_asli = tk.Frame(f, bg=PANEL)
        hdr_asli.pack(fill="x", padx=12, pady=(10, 6))
        tk.Frame(hdr_asli, bg=AKSEN, width=3, height=16).pack(side="left", padx=(0, 8))
        tk.Label(hdr_asli, text="Gambar Asli", font=F_JUDUL,
                 bg=PANEL, fg=PUTIH).pack(side="left")
        tk.Button(
            hdr_asli, text="⊡ Reset Zoom",
            command=self.canvas_asli.reset_zoom,
            font=F_KECIL, bg="#21262d", fg=TEKS2,
            activebackground="#30363d", activeforeground=TEKS,
            relief="flat", bd=0, padx=6, pady=2, cursor="hand2",
        ).pack(side="right")

        self._cvs_nama_asli = tk.Canvas(
            hdr_asli, bg=PANEL, height=18,
            highlightthickness=0, cursor="fleur",
        )
        self._cvs_nama_asli.pack(side="left", fill="x", expand=True, padx=(10, 4))
        self._inner_nama_asli = tk.Frame(self._cvs_nama_asli, bg=PANEL)
        self._cvs_nama_asli.create_window((0, 0), window=self._inner_nama_asli, anchor="nw")
        self.lbl_info_asli = tk.Label(
            self._inner_nama_asli, text="", font=F_KECIL, bg=PANEL, fg=TEKS2,
            cursor="fleur")
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
        self.canvas_prep.pack(fill="both", expand=True, padx=12, pady=(0, 4))

        self._badge = {}

    def _bangun_panel_kontrol_analisis(self, parent):
        f = tk.Frame(parent, bg=PANEL)
        f.grid(row=0, column=2, sticky="nsew", padx=(4, 0), pady=4)
        f.grid_propagate(False)

        judul_seksi(f, "Kontrol & Analisis")

        baris_ctrl = tk.Frame(f, bg=PANEL)
        baris_ctrl.pack(fill="x", padx=12, pady=(0, 6))
        tk.Label(baris_ctrl, text="Klik untuk aktif/nonaktif:",
                 font=F_KECIL, bg=PANEL, fg=TEKS2).pack(side="left")
        tk.Button(
            baris_ctrl, text="↺ Reset Teknik",
            command=self._reset_teknik,
            font=F_KECIL, bg="#21262d", fg=TEKS2,
            activebackground="#30363d", activeforeground=TEKS,
            relief="flat", bd=0, padx=6, pady=2, cursor="hand2",
        ).pack(side="right")

        self.frame_badge = tk.Frame(f, bg=PANEL)
        self.frame_badge.pack(padx=12, fill="x")

        self._badge = {}
        baris1 = tk.Frame(self.frame_badge, bg=PANEL)
        baris1.pack(fill="x", pady=(0, 4))
        baris2 = tk.Frame(self.frame_badge, bg=PANEL)
        baris2.pack(fill="x")

        teknik_baris1 = ["Grayscale", "CLAHE", "Adaptive Gaussian"]
        teknik_baris2 = ["Line Removal"]

        for t in teknik_baris1:
            btn = tk.Button(
                baris1, text=t, font=F_BADGE,
                bg=TEKS3, fg="#555",
                activebackground=TEKS3, activeforeground="#555",
                relief="flat", bd=0, padx=10, pady=5,
                cursor="arrow",
                highlightthickness=0,
            )
            btn.pack(side="left", padx=(0, 4))
            btn.configure(command=lambda nama=t: self._toggle_teknik(nama))
            self._badge[t] = btn

        for t in teknik_baris2:
            btn = tk.Button(
                baris2, text=t, font=F_BADGE,
                bg=TEKS3, fg="#555",
                activebackground=TEKS3, activeforeground="#555",
                relief="flat", bd=0, padx=10, pady=5,
                cursor="arrow",
                highlightthickness=0,
            )
            btn.pack(side="left", padx=(0, 4))
            btn.configure(command=lambda nama=t: self._toggle_teknik(nama))
            self._badge[t] = btn

        tk.Frame(f, bg=BORDER, height=1).pack(fill="x", padx=12, pady=12)

        tk.Label(f, text="Analisis Gambar",
                 font=F_JUDUL, bg=PANEL, fg=PUTIH).pack(padx=12, anchor="w", pady=(0, 8))

        hdr = tk.Frame(f, bg=PANEL)
        hdr.pack(fill="x", padx=12)
        hdr.columnconfigure(0, weight=3)
        hdr.columnconfigure(1, weight=2)
        hdr.columnconfigure(2, weight=2)

        tk.Label(hdr, text="Metrik",   font=F_KECIL, bg=PANEL, fg=TEKS2).grid(row=0, column=0, sticky="w")
        tk.Label(hdr, text="Sebelum",  font=F_KECIL, bg=PANEL, fg=TEKS2).grid(row=0, column=1, sticky="w")
        tk.Label(hdr, text="Sesudah",  font=F_KECIL, bg=PANEL, fg=TEKS2).grid(row=0, column=2, sticky="w")

        tk.Frame(f, bg=BORDER, height=1).pack(fill="x", padx=12, pady=4)

        self._analisis_labels = {}
        metrik_list = [
            ("intensitas",   "Intensitas\nCahaya",  "rata-rata kecerahan piksel (0-255)"),
            ("noise",        "Tingkat\nNoise",       "std residual blur Gaussian"),
            ("rasio_garis",  "Deteksi\nGaris",       "rasio piksel garis (%)"),
        ]

        for key, nama, _ in metrik_list:
            row_f = tk.Frame(f, bg=PANEL)
            row_f.pack(fill="x", padx=12, pady=4)
            row_f.columnconfigure(0, weight=3)
            row_f.columnconfigure(1, weight=2)
            row_f.columnconfigure(2, weight=2)

            tk.Label(row_f, text=nama, font=F_KECIL, bg=PANEL,
                     fg=TEKS, justify="left").grid(row=0, column=0, sticky="w")

            lbl_before = tk.Label(row_f, text="—", font=F_MONO,
                                  bg=CARD, fg=TEKS2, width=8, pady=4)
            lbl_before.grid(row=0, column=1, sticky="ew", padx=(4, 2))

            lbl_after  = tk.Label(row_f, text="—", font=F_MONO,
                                  bg=CARD, fg=TEKS2, width=8, pady=4)
            lbl_after.grid(row=0, column=2, sticky="ew", padx=(2, 0))

            self._analisis_labels[key] = (lbl_before, lbl_after)

    def _bangun_panel_gt(self, parent):
        f = tk.Frame(parent, bg=PANEL)
        f.grid(row=0, column=0, sticky="nsew", padx=(0, 4), pady=4)

        hdr_gt = tk.Frame(f, bg=PANEL)
        hdr_gt.pack(fill="x", padx=12, pady=(10, 6))
        tk.Frame(hdr_gt, bg=AKSEN, width=3, height=16).pack(side="left", padx=(0, 8))
        tk.Label(hdr_gt, text="Ground Truth", font=F_JUDUL,
                 bg=PANEL, fg=PUTIH).pack(side="left")

        self._cvs_nama_gt = tk.Canvas(
            hdr_gt, bg=PANEL, height=18,
            highlightthickness=0, cursor="fleur",
        )
        self._cvs_nama_gt.pack(side="left", fill="x", expand=True, padx=(10, 0))
        self._inner_nama_gt = tk.Frame(self._cvs_nama_gt, bg=PANEL)
        self._cvs_nama_gt.create_window((0, 0), window=self._inner_nama_gt, anchor="nw")

        self.lbl_gt_file = tk.Label(
            self._inner_nama_gt, text="", font=F_KECIL, bg=PANEL, fg=TEKS2,
            cursor="fleur")
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
        inner.pack(fill="both", expand=True, padx=12, pady=(4, 12))

        self.txt_gt = self._buat_textbox(inner, bg=CARD)
        self.txt_gt.pack(side="left", fill="both", expand=True)
        sb = ttk.Scrollbar(inner, orient="vertical", command=self.txt_gt.yview)
        sb.pack(side="right", fill="y")
        self.txt_gt.configure(yscrollcommand=sb.set)

    def _bangun_panel_ocr_asli(self, parent):
        f = tk.Frame(parent, bg=PANEL)
        f.grid(row=0, column=1, sticky="nsew", padx=4, pady=4)

        judul_seksi(f, "OCR Tanpa Preprocessing")

        self.frame_metrik_asli = tk.Frame(f, bg=PANEL)
        self.frame_metrik_asli.pack(fill="x", padx=12, pady=(0, 4))
        self._buat_kartu_metrik(self.frame_metrik_asli, 0, "CER", prefix="asli")
        self._buat_kartu_metrik(self.frame_metrik_asli, 1, "WER", prefix="asli")
        self.frame_metrik_asli.columnconfigure(0, weight=1)
        self.frame_metrik_asli.columnconfigure(1, weight=1)

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

        self.frame_metrik_prep = tk.Frame(f, bg=PANEL)
        self.frame_metrik_prep.pack(fill="x", padx=12, pady=(0, 4))
        self._buat_kartu_metrik(self.frame_metrik_prep, 0, "CER", prefix="prep")
        self._buat_kartu_metrik(self.frame_metrik_prep, 1, "WER", prefix="prep")
        self.frame_metrik_prep.columnconfigure(0, weight=1)
        self.frame_metrik_prep.columnconfigure(1, weight=1)

        inner = tk.Frame(f, bg=CARD)
        inner.pack(fill="both", expand=True, padx=12, pady=(0, 8))
        self.txt_ocr_prep = self._buat_textbox(inner, bg=CARD)
        self.txt_ocr_prep.pack(side="left", fill="both", expand=True)
        sb = ttk.Scrollbar(inner, orient="vertical", command=self.txt_ocr_prep.yview)
        sb.pack(side="right", fill="y")
        self.txt_ocr_prep.configure(yscrollcommand=sb.set)

    def _buat_textbox(self, parent, bg=CARD):
        return tk.Text(
            parent, font=F_MONO, bg=bg, fg=TEKS,
            insertbackground=TEKS, relief="flat", bd=0,
            wrap="word", padx=10, pady=10, state="disabled",
        )

    def _buat_kartu_metrik(self, parent, col, label, prefix):
        card = tk.Frame(parent, bg=CARD)
        card.grid(row=0, column=col, sticky="ew", padx=(0 if col == 0 else 3,
                                                      3 if col == 0 else 0))
        tk.Label(card, text=label, font=F_KECIL, bg=CARD, fg=TEKS2).pack(pady=(6, 0))
        lbl_val = tk.Label(card, text="—", font=("Segoe UI", 18, "bold"), bg=CARD, fg=AKSEN)
        lbl_val.pack()
        lbl_ind = tk.Label(card, text="", font=F_KECIL, bg=CARD, fg=TEKS2)
        lbl_ind.pack(pady=(0, 6))
        setattr(self, f"lbl_{prefix}_{label.lower()}", lbl_val)
        setattr(self, f"lbl_{prefix}_{label.lower()}_ind", lbl_ind)

    def _pilih_gambar(self):
        path = filedialog.askopenfilename(
            filetypes=[("Gambar", "*.png *.jpg *.jpeg *.bmp")]
        )
        if not path:
            return
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            messagebox.showerror("Error", "Gagal membaca gambar.")
            return
        self.img_asli = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        self._tampil_gambar(self.canvas_asli, self.img_asli, "_tk_asli")
        self.lbl_info_asli.configure(text=f"—  {os.path.basename(path)}")

    def _pilih_gt(self):
        path = filedialog.askopenfilename(
            filetypes=[("Teks", "*.txt"), ("Semua file", "*.*")]
        )
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

        self.configure(cursor="watch")
        self.update()

        img_gray = to_grayscale(self.img_asli)
        self.teks_ocr_asli = run_ocr(img_gray)
        self._isi_teks(self.txt_ocr_asli, self.teks_ocr_asli)

        self.img_prep, self.log_teknik = pipeline_adaptive(self.img_asli)
        self._tampil_gambar(self.canvas_prep, self.img_prep, gray=True)
        self.teks_ocr_prep = run_ocr(self.img_prep)
        self._isi_teks(self.txt_ocr_prep, self.teks_ocr_prep)

        gray_asli = to_grayscale(self.img_asli)
        a_before = analisis_gambar(gray_asli)
        a_after = analisis_gambar(self.img_prep)
        self._update_analisis(a_before, a_after)

        self.aktif_teknik = {
            "Grayscale": True,
            "CLAHE": bool(self.log_teknik.get("clahe")),
            "Adaptive Gaussian": True,
            "Line Removal": bool(self.log_teknik.get("line removal")),
        }
        self.aktif_awal = dict(self.aktif_teknik)
        self.dimatikan.clear()
        self._perbarui_badge()

        if self.teks_gt.strip():
            m_asli = hitung_cer_wer(self.teks_ocr_asli, self.teks_gt)
            m_prep = hitung_cer_wer(self.teks_ocr_prep, self.teks_gt)
            self._update_metrik(m_asli, m_prep)
        else:
            for attr in ["lbl_asli_cer", "lbl_asli_wer", "lbl_prep_cer", "lbl_prep_wer"]:
                getattr(self, attr).configure(text="—", fg=TEKS2)
            for attr in ["lbl_asli_cer_ind", "lbl_asli_wer_ind",
                         "lbl_prep_cer_ind", "lbl_prep_wer_ind"]:
                getattr(self, attr).configure(text="Upload GT untuk evaluasi")

        self.configure(cursor="")

    def _update_metrik(self, m_asli: dict, m_prep: dict):
        def format_val(v):
            return f"{v:.2f}%" if v is not None else "—"

        def indikator(v_asli, v_prep):
            if v_asli is None or v_prep is None:
                return "", TEKS2
            selisih = v_prep - v_asli
            if abs(selisih) < 0.01:
                return "= tidak berubah", TEKS2
            elif selisih < 0:
                return f"▼ {abs(selisih):.2f}% (membaik)", HIJAU
            else:
                return f"▲ {selisih:.2f}% (memburuk)", MERAH

        self.lbl_asli_cer.configure(
            text=format_val(m_asli["cer"]),
            fg=KUNING if m_asli["cer"] is not None else TEKS2
        )
        self.lbl_prep_cer.configure(
            text=format_val(m_prep["cer"]),
            fg=HIJAU if (m_prep["cer"] or 999) < (m_asli["cer"] or 999) else
               MERAH if (m_prep["cer"] or 0) > (m_asli["cer"] or 0) else TEKS2
        )
        teks_ind, warna_ind = indikator(m_asli["cer"], m_prep["cer"])
        self.lbl_asli_cer_ind.configure(text="sebelum preprocessing")
        self.lbl_prep_cer_ind.configure(text=teks_ind, fg=warna_ind)

        self.lbl_asli_wer.configure(
            text=format_val(m_asli["wer"]),
            fg=KUNING if m_asli["wer"] is not None else TEKS2
        )
        self.lbl_prep_wer.configure(
            text=format_val(m_prep["wer"]),
            fg=HIJAU if (m_prep["wer"] or 999) < (m_asli["wer"] or 999) else
               MERAH if (m_prep["wer"] or 0) > (m_asli["wer"] or 0) else TEKS2
        )
        teks_ind, warna_ind = indikator(m_asli["wer"], m_prep["wer"])
        self.lbl_asli_wer_ind.configure(text="sebelum preprocessing")
        self.lbl_prep_wer_ind.configure(text=teks_ind, fg=warna_ind)

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
            self._perbarui_badge()
            self.update()
        else:
            if nama == "Line Removal" and not self.aktif_teknik.get("Adaptive Gaussian", False):
                return
            if nama != "Grayscale" and not self.aktif_teknik.get("Grayscale", False):
                return
            self.aktif_teknik[nama] = True

        self._perbarui_badge()
        self.update()
        self._rerun_prep()

    def _reset_teknik(self):
        if not self.aktif_awal:
            return
        self.aktif_teknik = dict(self.aktif_awal)
        self._rerun_prep()

    def _rerun_prep(self):
        if self.img_asli is None or not self.log_teknik:
            return

        self.configure(cursor="watch")
        self.update()

        img_tampil, img_ocr = pipeline_manual(self.img_asli, self.aktif_teknik)
        self.img_prep = img_tampil

        gray_mode = self.aktif_teknik.get("Grayscale", True)
        self._tampil_gambar(self.canvas_prep, img_tampil, gray=gray_mode)

        self.teks_ocr_prep = run_ocr(img_ocr)
        self._isi_teks(self.txt_ocr_prep, self.teks_ocr_prep)

        if self.teks_gt.strip():
            m_asli = hitung_cer_wer(self.teks_ocr_asli, self.teks_gt)
            m_prep = hitung_cer_wer(self.teks_ocr_prep, self.teks_gt)
            self._update_metrik(m_asli, m_prep)

        a_after = analisis_gambar(img_ocr)

        lbl_b_int, _ = self._analisis_labels["intensitas"]
        lbl_b_noise, _ = self._analisis_labels["noise"]
        lbl_b_garis, _ = self._analisis_labels["rasio_garis"]
        try:
            a_before = {
                "intensitas": float(lbl_b_int.cget("text")),
                "noise": float(lbl_b_noise.cget("text")),
                "rasio_garis": float(lbl_b_garis.cget("text")),
            }
        except Exception:
            a_before = analisis_gambar(to_grayscale(self.img_asli))

        self._update_analisis(a_before, a_after)
        self._perbarui_badge()
        self.configure(cursor="")

    def _perbarui_badge(self):
        for nama, btn in self._badge.items():
            if not self.aktif_teknik:
                btn.configure(
                    bg=TEKS3, fg="#555",
                    activebackground=TEKS3, activeforeground="#555",
                    cursor="arrow", text=nama,
                )
                continue

            aktif = self.aktif_teknik.get(nama, False)
            gray_aktif = self.aktif_teknik.get("Grayscale", False)
            bin_aktif  = self.aktif_teknik.get("Adaptive Gaussian", False)


            if nama == "grayscale":
                bisa_klik = True
            elif nama == "line_removal":
                bisa_klik = aktif or (gray_aktif and bin_aktif)
            else:
                bisa_klik = aktif or gray_aktif

            if aktif:
                btn.configure(
                    bg=AKSEN, fg=PUTIH,
                    activebackground=AKSEN, activeforeground=PUTIH,
                    relief="flat",
                    cursor="hand2" if bisa_klik else "arrow",
                    text=nama,
                )
            else:
                btn.configure(
                    bg=TEKS3, fg="#888",
                    activebackground=TEKS3, activeforeground="#888",
                    relief="flat",
                    cursor="hand2" if bisa_klik else "arrow",
                    text=nama,
                )

    def _update_analisis(self, before: dict, after: dict):
        def warna_noise(v):
            if v < 5:   return HIJAU
            if v < 15:  return KUNING
            return MERAH

        def warna_intensitas(v):
            if 100 <= v <= 200: return HIJAU
            if 80  <= v <= 220: return KUNING
            return MERAH

        # Tidak menampilkan label "Ada Garis?".
        # Warna rasio_garis menggunakan ambang yang sama dengan analisis_gambar.
        def warna_rasio(v):
            return MERAH if v > 0.3 else HIJAU

        for key, (lbl_b, lbl_a) in self._analisis_labels.items():
            val_b = before.get(key, 0)
            val_a = after.get(key, 0)
            lbl_b.configure(text=f"{val_b:.2f}")
            lbl_a.configure(text=f"{val_a:.2f}")

            if key == "noise":
                lbl_b.configure(fg=warna_noise(val_b))
                lbl_a.configure(fg=warna_noise(val_a))
            elif key == "intensitas":
                lbl_b.configure(fg=warna_intensitas(val_b))
                lbl_a.configure(fg=warna_intensitas(val_a))
            elif key == "rasio_garis":
                lbl_b.configure(fg=warna_rasio(val_b))
                lbl_a.configure(fg=warna_rasio(val_a))

    def _bangun_scroll_container(self):
        wrap = tk.Frame(self, bg=BG)
        wrap.pack(fill="both", expand=True)

        self._scrollbar = ttk.Scrollbar(wrap, orient="vertical")
        self._scrollbar.pack(side="right", fill="y")

        self._scroll_canvas = tk.Canvas(
            wrap, bg=BG, highlightthickness=0,
            yscrollcommand=self._scrollbar.set,
        )
        self._scroll_canvas.pack(side="left", fill="both", expand=True)
        self._scrollbar.configure(command=self._scroll_canvas.yview)

        self._scroll_inner = tk.Frame(self._scroll_canvas, bg=BG)
        self._scroll_win_id = self._scroll_canvas.create_window(
            (0, 0), window=self._scroll_inner, anchor="nw"
        )

        self._scroll_inner.bind("<Configure>", self._on_scroll_inner_configure)
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
        if isinstance(event.widget, ZoomableCanvas):
            return
        self._scroll_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _on_scroll_up(self, event):
        self._scroll_canvas.yview_scroll(-1, "units")

    def _on_scroll_down(self, event):
        self._scroll_canvas.yview_scroll(1, "units")

    def _drag_nama_asli(self, event):
        dx = self._cvs_nama_asli._dx - event.x
        self._cvs_nama_asli.xview_scroll(int(dx / 2), "units")
        self._cvs_nama_asli._dx = event.x

    def _drag_nama_gt(self, event):
        dx = self._cvs_nama_gt._dx - event.x
        self._cvs_nama_gt.xview_scroll(int(dx / 2), "units")
        self._cvs_nama_gt._dx = event.x

    def _tampil_gambar(self, canvas: ZoomableCanvas, img, attr=None, gray=False):
        canvas.set_image(img, gray=gray)

    def _isi_teks(self, widget, teks):
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", teks)
        widget.configure(state="disabled")

    def _salin(self, widget):
        widget.configure(state="normal")
        teks = widget.get("1.0", "end").strip()
        widget.configure(state="disabled")
        if teks:
            self.clipboard_clear()
            self.clipboard_append(teks)
            messagebox.showinfo("Berhasil", "Teks berhasil disalin ke clipboard.")

    def _reset(self):
        self.img_asli = self.img_prep = None
        self.teks_gt = self.teks_ocr_asli = self.teks_ocr_prep = ""
        self.log_teknik = {}
        self.canvas_asli.clear()
        self.canvas_prep.clear()
        self.lbl_info_asli.configure(text="")
        self.lbl_gt_file.configure(text="")

        for w in [self.txt_gt, self.txt_ocr_asli, self.txt_ocr_prep]:
            self._isi_teks(w, "")

        for attr in ["lbl_asli_cer", "lbl_asli_wer", "lbl_prep_cer", "lbl_prep_wer"]:
            getattr(self, attr).configure(text="—", fg=AKSEN)
        for attr in ["lbl_asli_cer_ind", "lbl_asli_wer_ind",
                     "lbl_prep_cer_ind", "lbl_prep_wer_ind"]:
            getattr(self, attr).configure(text="", fg=TEKS2)

        for key, (lbl_b, lbl_a) in self._analisis_labels.items():
            lbl_b.configure(text="—", fg=TEKS2)
            lbl_a.configure(text="—", fg=TEKS2)

        self.dimatikan.clear()
        self.aktif_teknik.clear()
        self.aktif_awal.clear()
        for nama, btn in self._badge.items():
            btn.configure(
                text=nama, bg=TEKS3, fg="#555",
                activebackground=TEKS3, activeforeground="#555",
                cursor="arrow"
            )


if __name__ == "__main__":
    app = App()
    app.mainloop()

