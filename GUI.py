"""
gui_ocr.py
==========
GUI Preprocessing Adaptif OCR — Revisi untuk Laporan Tugas Akhir
Fitur:
  - Upload gambar dan ground truth (.txt) secara manual
  - Tampilan gambar asli vs hasil preprocessing
  - OCR tanpa preprocessing vs dengan preprocessing (adaptif)
  - Tabel perbandingan CER & WER (sebelum vs sesudah)
  - Ground truth ditampilkan berdampingan untuk perbandingan langsung
"""

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


def denoise(gray: np.ndarray) -> tuple[np.ndarray, bool]:
    noise = float(np.std(gray - cv2.GaussianBlur(gray, (5, 5), 0)))
    if noise <= 10:
        return gray, False
    return cv2.medianBlur(gray, 3), True


def clahe(gray: np.ndarray) -> tuple[np.ndarray, bool]:
    if float(np.std(gray)) >= 30:
        return gray, False
    enh = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return enh.apply(gray), True


def binarize(gray: np.ndarray) -> np.ndarray:
    gray = cv2.medianBlur(gray, 3)
    return cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 111, 50
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
    if np.count_nonzero(mask) / (h * w) <= 0.005:
        return binary, False
    mask  = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
    hasil = cv2.bitwise_and(inv, inv, mask=cv2.bitwise_not(mask))
    return ~hasil, True


def pipeline_adaptive(image: np.ndarray) -> tuple[np.ndarray, dict]:
    log = {}
    img = to_grayscale(image);          log["grayscale"]    = True
    img, log["denoise"]      = denoise(img)
    img, log["clahe"]        = clahe(img)
    img = binarize(img);                log["binarize"]     = True
    img, log["line_removal"] = remove_lines(img)
    return img, log


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
    text = text.replace("|", "").replace("—", "-").replace("_", "-")
    text = text.replace("√", "v").replace("✓", "v")
    text = text.encode("ascii", errors="ignore").decode("ascii")
    text = re.sub(r'[ \t]+', ' ', text)
    text = "\n".join(l.strip() for l in text.splitlines())
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
    """
    Canvas dengan fitur zoom (scroll mouse) dan pan (drag).
    - Scroll roda mouse   : zoom in / zoom out
    - Klik kiri + drag    : geser (pan) gambar
    - Tombol reset        : kembali ke fit-to-canvas
    Zoom dibatasi antara 0.1× hingga 20×.
    """

    ZOOM_STEP   = 1.25   # faktor per satu klik scroll
    ZOOM_MIN    = 0.1
    ZOOM_MAX    = 20.0

    def __init__(self, parent, **kw):
        kw.setdefault("bg", "#0a0d14")
        kw.setdefault("highlightthickness", 0)
        kw.setdefault("cursor", "crosshair")
        super().__init__(parent, **kw)

        self._pil_src  : Image.Image | None = None  # gambar asli (PIL) full-res
        self._tk_img   : ImageTk.PhotoImage | None = None
        self._zoom     : float = 1.0   # skala saat ini relatif terhadap fit
        self._fit_scale: float = 1.0   # skala fit-to-canvas
        self._offset_x : float = 0.0   # offset tengah canvas → titik tampil
        self._offset_y : float = 0.0

        # Drag state
        self._drag_start: tuple[int, int] | None = None

        self.bind("<Configure>",       self._on_resize)
        self.bind("<MouseWheel>",      self._on_scroll_win)   # Windows / macOS
        self.bind("<Button-4>",        self._on_scroll_up)    # Linux scroll up
        self.bind("<Button-5>",        self._on_scroll_down)  # Linux scroll down
        self.bind("<ButtonPress-1>",   self._on_drag_start)
        self.bind("<B1-Motion>",       self._on_drag_move)
        self.bind("<ButtonRelease-1>", self._on_drag_end)



    # ── Muat gambar baru ──────────────────────────────────────────────────────

    def set_image(self, img: np.ndarray, gray: bool = False):
        """Terima np.ndarray (RGB atau grayscale) dan reset zoom ke fit."""
        if gray and len(img.shape) == 2:
            self._pil_src = Image.fromarray(img).convert("RGB")
        else:
            self._pil_src = Image.fromarray(img)
        self._reset_zoom()

    def clear(self):
        self._pil_src = None
        self._tk_img  = None
        self.delete("all")


    # ── Reset ke fit ─────────────────────────────────────────────────────────

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

    # ── Render ────────────────────────────────────────────────────────────────

    def _render(self):
        if self._pil_src is None:
            return

        self.update_idletasks()
        cw = self.winfo_width()  or 500
        ch = self.winfo_height() or 400

        scale  = self._fit_scale * self._zoom
        iw, ih = self._pil_src.size
        nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))

        resized     = self._pil_src.resize((nw, nh), Image.LANCZOS)
        self._tk_img = ImageTk.PhotoImage(resized)

        self.delete("img")
        self.create_image(
            self._offset_x, self._offset_y,
            image=self._tk_img, anchor="center", tags="img"
        )


    # ── Event handlers ────────────────────────────────────────────────────────

    def _on_resize(self, event):
        if self._pil_src is not None:
            # Pertahankan zoom, hanya recalc fit_scale
            iw, ih = self._pil_src.size
            self._fit_scale = min(event.width / iw, event.height / ih, 1.0)
            # Geser offset supaya tetap di tengah jika baru pertama
            self._offset_x = event.width  / 2
            self._offset_y = event.height / 2
            self._render()

    def _zoom_at(self, x: int, y: int, faktor: float):
        """Zoom in/out dengan pusat di posisi kursor (x, y)."""
        new_zoom = max(self.ZOOM_MIN, min(self.ZOOM_MAX, self._zoom * faktor))
        ratio    = new_zoom / self._zoom
        # Geser offset supaya titik di bawah kursor tetap diam
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

        self._bangun_ui()

    # ── Layout utama ──────────────────────────────────────────────────────────

    def _bangun_ui(self):
        self._bangun_header()

        # Baris atas: gambar (60% tinggi)
        self.frame_atas = tk.Frame(self, bg=BG)
        self.frame_atas.pack(fill="both", expand=True, padx=12, pady=(0, 4))
        self.frame_atas.columnconfigure(0, weight=1)
        self.frame_atas.columnconfigure(1, weight=1)
        self.frame_atas.rowconfigure(0, weight=1)

        self._bangun_panel_gambar_asli(self.frame_atas)
        self._bangun_panel_gambar_prep(self.frame_atas)

        # Baris bawah: teks & evaluasi
        self.frame_bawah = tk.Frame(self, bg=BG)
        self.frame_bawah.pack(fill="both", expand=True, padx=12, pady=(0, 10))
        self.frame_bawah.columnconfigure(0, weight=1)
        self.frame_bawah.columnconfigure(1, weight=1)
        self.frame_bawah.columnconfigure(2, weight=1)
        self.frame_bawah.rowconfigure(0, weight=1)

        self._bangun_panel_gt(self.frame_bawah)
        self._bangun_panel_ocr_asli(self.frame_bawah)
        self._bangun_panel_ocr_prep(self.frame_bawah)

    # ── Header ────────────────────────────────────────────────────────────────

    def _bangun_header(self):
        hdr = tk.Frame(self, bg=PANEL, height=58)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)

        # Kiri: judul
        kiri = tk.Frame(hdr, bg=PANEL)
        kiri.pack(side="left", padx=20, pady=10)
        tk.Frame(kiri, bg=AKSEN, width=4, height=30).pack(side="left", padx=(0, 10))
        tk.Label(kiri, text="Evaluasi Preprocessing Adaptif OCR",
                 font=("Segoe UI", 13, "bold"), bg=PANEL, fg=PUTIH).pack(side="left")
        tk.Label(kiri, text="  —  Alat Bantu Penulisan Laporan Tugas Akhir",
                 font=F_LABEL, bg=PANEL, fg=TEKS2).pack(side="left")

        # Kanan: tombol aksi
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

    # ── Panel gambar asli ─────────────────────────────────────────────────────

    def _bangun_panel_gambar_asli(self, parent):
        f = tk.Frame(parent, bg=PANEL)
        f.grid(row=0, column=0, sticky="nsew", padx=(0, 4), pady=4)

        self.canvas_asli = ZoomableCanvas(f)
        judul_seksi(f, "Gambar Asli", canvas_zoom=self.canvas_asli)
        self.canvas_asli.pack(fill="both", expand=True, padx=12, pady=(0, 4))

        self.lbl_info_asli = tk.Label(
            f, text="Belum ada gambar", font=F_KECIL, bg=PANEL, fg=TEKS2)
        self.lbl_info_asli.pack(pady=(0, 8))

    # ── Panel gambar preprocessing ────────────────────────────────────────────

    def _bangun_panel_gambar_prep(self, parent):
        f = tk.Frame(parent, bg=PANEL)
        f.grid(row=0, column=1, sticky="nsew", padx=(4, 0), pady=4)

        self.canvas_prep = ZoomableCanvas(f)
        judul_seksi(f, "Hasil Preprocessing Adaptif", warna_aksen=AKSEN2,
                    canvas_zoom=self.canvas_prep)
        self.canvas_prep.pack(fill="both", expand=True, padx=12, pady=(0, 4))

        # Badge teknik
        self.frame_badge = tk.Frame(f, bg=PANEL)
        self.frame_badge.pack(pady=(0, 8))
        self._badge = {}
        for t in ["grayscale", "denoise", "clahe", "binarize", "line_removal"]:
            lbl = tk.Label(self.frame_badge, text=t, font=F_BADGE,
                           bg=TEKS3, fg="#555", padx=6, pady=2)
            lbl.pack(side="left", padx=2)
            self._badge[t] = lbl

    # ── Panel ground truth ────────────────────────────────────────────────────

    def _bangun_panel_gt(self, parent):
        f = tk.Frame(parent, bg=PANEL)
        f.grid(row=0, column=0, sticky="nsew", padx=(0, 4), pady=4)

        judul_seksi(f, "Ground Truth")

        self.lbl_gt_file = tk.Label(
            f, text="Belum diupload", font=F_KECIL, bg=PANEL, fg=TEKS2)
        self.lbl_gt_file.pack(padx=12, anchor="w")

        inner = tk.Frame(f, bg=CARD)
        inner.pack(fill="both", expand=True, padx=12, pady=(4, 12))

        self.txt_gt = self._buat_textbox(inner, bg=CARD)
        self.txt_gt.pack(side="left", fill="both", expand=True)
        sb = ttk.Scrollbar(inner, orient="vertical", command=self.txt_gt.yview)
        sb.pack(side="right", fill="y")
        self.txt_gt.configure(yscrollcommand=sb.set)

    # ── Panel OCR tanpa preprocessing ────────────────────────────────────────

    def _bangun_panel_ocr_asli(self, parent):
        f = tk.Frame(parent, bg=PANEL)
        f.grid(row=0, column=1, sticky="nsew", padx=4, pady=4)

        judul_seksi(f, "OCR Tanpa Preprocessing")

        # Kartu CER/WER (sebelum)
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

        buat_tombol(f, "Salin", lambda: self._salin(self.txt_ocr_asli),
                    warna="#21262d", lebar=10).pack(pady=(0, 8))

    # ── Panel OCR dengan preprocessing ───────────────────────────────────────

    def _bangun_panel_ocr_prep(self, parent):
        f = tk.Frame(parent, bg=PANEL)
        f.grid(row=0, column=2, sticky="nsew", padx=(4, 0), pady=4)

        judul_seksi(f, "OCR Setelah Preprocessing", warna_aksen=AKSEN2)

        # Kartu CER/WER (sesudah)
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

        buat_tombol(f, "Salin", lambda: self._salin(self.txt_ocr_prep),
                    warna="#21262d", lebar=10).pack(pady=(0, 8))

    # ── Helpers UI ────────────────────────────────────────────────────────────

    def _buat_textbox(self, parent, bg=CARD):
        return tk.Text(
            parent, font=F_MONO, bg=bg, fg=TEKS,
            insertbackground=TEKS, relief="flat", bd=0,
            wrap="word", padx=10, pady=10, state="disabled",
        )

    def _buat_kartu_metrik(self, parent, col, label, prefix):
        """Buat satu kartu metrik CER atau WER dan simpan referensinya."""
        card = tk.Frame(parent, bg=CARD)
        card.grid(row=0, column=col, sticky="ew", padx=(0 if col else 0, 3 if col == 0 else 0))
        if col == 0:
            card.grid(padx=(0, 3))
        else:
            card.grid(padx=(3, 0))

        tk.Label(card, text=label, font=F_KECIL, bg=CARD, fg=TEKS2).pack(pady=(6, 0))

        lbl_val = tk.Label(card, text="—", font=("Segoe UI", 18, "bold"), bg=CARD, fg=AKSEN)
        lbl_val.pack()

        lbl_ind = tk.Label(card, text="", font=F_KECIL, bg=CARD, fg=TEKS2)
        lbl_ind.pack(pady=(0, 6))

        # Simpan referensi
        setattr(self, f"lbl_{prefix}_{label.lower()}", lbl_val)
        setattr(self, f"lbl_{prefix}_{label.lower()}_ind", lbl_ind)

    # ── Aksi tombol ───────────────────────────────────────────────────────────

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
        h, w = self.img_asli.shape[:2]
        self.lbl_info_asli.configure(
            text=f"{os.path.basename(path)}  —  {w}×{h} px")

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
            text=f"✔  {os.path.basename(path)}  ({len(self.teks_gt)} karakter)")
        self._isi_teks(self.txt_gt, self.teks_gt)

    def _jalankan(self):
        if self.img_asli is None:
            messagebox.showinfo("Info", "Pilih gambar terlebih dahulu.")
            return

        self.configure(cursor="watch")
        self.update()

        # ── OCR tanpa preprocessing ──────────────────────────────────────────
        img_gray = to_grayscale(self.img_asli)
        # Konversi ke RGB agar Tesseract bisa baca (atau tetap gray)
        self.teks_ocr_asli = run_ocr(img_gray)
        self._isi_teks(self.txt_ocr_asli, self.teks_ocr_asli)

        # ── OCR dengan preprocessing adaptif ────────────────────────────────
        self.img_prep, self.log_teknik = pipeline_adaptive(self.img_asli)
        self._tampil_gambar(self.canvas_prep, self.img_prep, "_tk_prep", gray=True)
        self.teks_ocr_prep = run_ocr(self.img_prep)
        self._isi_teks(self.txt_ocr_prep, self.teks_ocr_prep)

        # ── Badge teknik ─────────────────────────────────────────────────────
        for nama, aktif in self.log_teknik.items():
            if nama in self._badge:
                self._badge[nama].configure(
                    bg=AKSEN if aktif else TEKS3,
                    fg=PUTIH if aktif else "#555"
                )

        # ── Hitung & tampilkan CER/WER ───────────────────────────────────────
        if self.teks_gt.strip():
            m_asli = hitung_cer_wer(self.teks_ocr_asli, self.teks_gt)
            m_prep = hitung_cer_wer(self.teks_ocr_prep, self.teks_gt)
            self._update_metrik(m_asli, m_prep)
        else:
            # Ground truth belum diupload — tampilkan tanda tanya
            for attr in ["lbl_asli_cer", "lbl_asli_wer", "lbl_prep_cer", "lbl_prep_wer"]:
                getattr(self, attr).configure(text="—", fg=TEKS2)
            for attr in ["lbl_asli_cer_ind", "lbl_asli_wer_ind",
                         "lbl_prep_cer_ind", "lbl_prep_wer_ind"]:
                getattr(self, attr).configure(text="Upload GT untuk evaluasi")

        self.configure(cursor="")

    def _update_metrik(self, m_asli: dict, m_prep: dict):
        """Isi nilai CER/WER dan indikator naik/turun."""
        def format_val(v):
            return f"{v:.2f}%" if v is not None else "—"

        def indikator(v_asli, v_prep):
            """Kembalikan (teks, warna) perubahan CER/WER."""
            if v_asli is None or v_prep is None:
                return "", TEKS2
            selisih = v_prep - v_asli
            if abs(selisih) < 0.01:
                return "= tidak berubah", TEKS2
            elif selisih < 0:
                return f"▼ {abs(selisih):.2f}% (membaik)", HIJAU
            else:
                return f"▲ {selisih:.2f}% (memburuk)", MERAH

        # CER
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

        # WER
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

    # ── Utility tampilan ──────────────────────────────────────────────────────

    def _tampil_gambar(self, canvas: ZoomableCanvas, img, attr=None, gray=False):
        """Tampilkan gambar di ZoomableCanvas (zoom & pan otomatis aktif)."""
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
        self.lbl_info_asli.configure(text="Belum ada gambar")
        self.lbl_gt_file.configure(text="Belum diupload")

        for w in [self.txt_gt, self.txt_ocr_asli, self.txt_ocr_prep]:
            self._isi_teks(w, "")

        for attr in ["lbl_asli_cer", "lbl_asli_wer", "lbl_prep_cer", "lbl_prep_wer"]:
            getattr(self, attr).configure(text="—", fg=AKSEN)
        for attr in ["lbl_asli_cer_ind", "lbl_asli_wer_ind",
                     "lbl_prep_cer_ind", "lbl_prep_wer_ind"]:
            getattr(self, attr).configure(text="", fg=TEKS2)

        for lbl in self._badge.values():
            lbl.configure(bg=TEKS3, fg="#555")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    app = App()
    app.mainloop()