import PyPDF2
from PyPDF2 import PdfReader

def baca_pdf(location):
    # 'rb' (Read Binary) wajib digunakan karena PDF adalah file biner, bukan teks biasa
    with open(location, "rb") as f:
        # Membuat objek reader untuk membedah struktur PDF
        reader = PdfReader(f)
        teks = ""
        # Melakukan perulangan untuk mengambil teks dari setiap halaman
        for halaman in reader.pages: # reader.pages (list yang berisi semua halaman pdf
            # extract_text() mengambil teks mentah dari halaman tersebut
            teks += halaman.extract_text()

    return teks

# Contoh penggunaan:
# Masukkan path file PDF kamu di sini
path_pdf = "/storage/emulated/0/Download/IDN BROKEN STRINGS.pdf"

try:
    hasil_teks = baca_pdf(path_pdf)
    
    print("=== HASIL EKSTRAKSI ===")
    # [:500] digunakan agar output di terminal tidak terlalu panjang
    print(hasil_teks[:100000]) 
except FileNotFoundError:
    print(f"Error: File tidak ditemukan di {path_pdf}")
except Exception as e:
    print(f"Terjadi kesalahan: {e}")

"""
PENJELASAN SYNTAX:
1. import PyPDF2: Mengambil alat untuk mengolah PDF.
2. open(location, "rb"): Membuka file dalam mode biner agar data PDF tidak rusak.
3. PdfReader: Mesin yang membaca isi file PDF yang sudah dibuka.
4. reader.pages: Kumpulan semua halaman yang ada di dalam PDF.
5. extract_text(): Mengambil tulisan yang ada di halaman menjadi teks string Python.
"""
