import os
import fitz
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

INPUT_PDF_DIR = "data/PDF"
OUTPUT_IMAGE_DIR = "data/gambar"

os.makedirs(INPUT_PDF_DIR, exist_ok=True)
os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)


def pixmap_to_bgr(pix):
    data = np.frombuffer(pix.samples, dtype=np.uint8)

    if pix.n == 1:
        arr = data.reshape((pix.height, pix.width))
        arr = np.stack([arr] * 3, axis=-1)
    else:
        arr = data.reshape((pix.height, pix.width, pix.n))
        if pix.n == 4:
            arr = arr[:, :, :3]

    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def pdf_to_images(pdf_path, dpi=300, log_callback=None):
    doc = fitz.open(pdf_path)
    base = os.path.splitext(os.path.basename(pdf_path))[0]
    total_pages = len(doc)

    for i, page in enumerate(doc):
        pix = page.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72))
        bgr = pixmap_to_bgr(pix)

        out_name = f"{base}_page{i + 1}.png"
        out_path = os.path.join(OUTPUT_IMAGE_DIR, out_name)
        cv2.imwrite(out_path, bgr)

        if log_callback:
            log_callback(f"Page {i + 1}/{total_pages} berhasil disimpan")

    doc.close()


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("PDF to Image Converter")
        self.geometry("500x300")
        self.resizable(False, False)
        self.pdf_paths = []
        self.build_ui()

    def build_ui(self):
        frame = ttk.Frame(self, padding=15)
        frame.pack(fill="both", expand=True)

        ttk.Label(
            frame,
            text="Konversi PDF ke Image (PNG)",
            font=("Arial", 12, "bold")
        ).pack(pady=5)

        ttk.Button(
            frame,
            text="Pilih File PDF",
            command=self.select_pdf
        ).pack(fill="x", pady=5)

        self.lbl_file = ttk.Label(frame, text="Belum ada file dipilih")
        self.lbl_file.pack()

        self.btn_convert = ttk.Button(
            frame,
            text="Convert ke Image",
            command=self.convert_pdf
        )
        self.btn_convert.pack(fill="x", pady=15)

        ttk.Label(frame, text="Log Proses:").pack(anchor="w")

        self.txt_log = tk.Text(frame, height=6)
        self.txt_log.pack(fill="both", expand=True)

    def select_pdf(self):
        paths = filedialog.askopenfilenames(
            initialdir=INPUT_PDF_DIR,
            filetypes=[("PDF Files", "*.pdf")]
        )

        if paths:
            self.pdf_paths = list(paths)
            if len(paths) == 1:
                self.lbl_file.config(text=os.path.basename(paths[0]))
            else:
                self.lbl_file.config(text=f"{len(paths)} file dipilih")

    def log(self, message):
        self.txt_log.insert("end", message + "\n")
        self.txt_log.see("end")
        self.update()

    def convert_pdf(self):
        if not self.pdf_paths:
            messagebox.showinfo("Info", "Silakan pilih file PDF terlebih dahulu.")
            return

        DPI = 300

        self.btn_convert.config(state="disabled")
        self.txt_log.delete("1.0", "end")
        self.log("Memulai konversi...")

        try:
            for pdf_path in self.pdf_paths:
                self.log(f"Memproses: {os.path.basename(pdf_path)}")
                pdf_to_images(pdf_path, dpi=DPI, log_callback=self.log)

            self.log("Konversi selesai.")
            messagebox.showinfo("Sukses", "Semua halaman berhasil dikonversi.")

        except Exception as e:
            messagebox.showerror("Error", str(e))

        finally:
            self.btn_convert.config(state="normal")


if __name__ == "__main__":
    app = App()
    app.mainloop()