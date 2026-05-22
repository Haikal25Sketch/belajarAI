import json
import PyPDF2
from PyPDF2 import PdfReader
import os
from dotenv import load_dotenv
import requests
def load_data(location):
    if not os.path.exists(location):
        print (f"FILE {location} TIDAK ADA!!!.")
        return []
    try:
        if location.endswith(".txt"):
            with open(location,"r")as f:
                lines =[line.strip() for line in f if line.strip()]
                print ("DATA TXT BERHASIL DIMUAT...")
            return lines
        elif location.endswith(".pdf"):
            with open(location,"rb") as f:
                reader = PdfReader(f)
                file = []
                for halaman in reader.pages:
                    teks_halaman = halaman.extract_text()
                    if teks_halaman:
                        file.append(teks_halaman)
                return file
    except (FileNotFoundError,PermissionError) as e:
        print (f"ERROR FILE : {e}!!!")

def potong(teks, ukuran=100, overlap=20): # Ukuran default digedein biar AI lebih pinter
    kata = teks.split() # pecah string jadi list
    hasil = []
    langkah = ukuran - overlap
    for i in range(0, len(kata), langkah):
        potongan = kata[i:i+ukuran]
        hasil.append(" ".join(potongan))
        if i + ukuran >= len(kata):
            break
    return hasil

load_dotenv()
token = os.getenv("HUGGINGFACE_TOKEN")
url = "https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L6-v2/pipeline/feature-extraction"
headers = {
    "Authorization":f"Bearer {token}",
    "Content-Type":"application/json"
}

# Fungsi dapetin embeddings (bisa terima satu teks atau list teks)
def get_embeddings(text):
    payload = {"inputs": text}
    try:
        # Menambahkan timeout agar tidak gantung
        response = requests.post(url, headers=headers, json=payload, timeout=15)
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 429:
            print ("EMBEDDINGS GAGAL | Status: 429 - Rate limit tercapai. Silakan tunggu.")
            return None
        else:
            print (f"EMBEDDINGS GAGAL | STATUS: {response.status_code} - {response.text}")
            return None

    except requests.exceptions.Timeout:
        print ("ERROR API: Waktu permintaan habis (Timeout).")
        return None
    except requests.exceptions.ConnectionError:
        print ("ERROR API: Gagal terhubung ke server. Periksa internet.")
        return None
    except Exception as e:
        logger.error(f"ERROR API TIDAK TERDUGA: {e}")
        return None
PATH_FILE = "/storage/emulated/0/Download/IDN BROKEN STRINGS.pdf"
NAMA_DB = f"Database_IDN BROKEN STRINGS.json"

if not os.path.exists(NAMA_DB):
    print ("Membuat database baru...")
    data_mentah = load_data(PATH_FILE)
    print()
    print (data_mentah)
    # 1. Kumpulkan semua chunk dulu
    semua_chunk = []
    for teks in data_mentah:
        chunks = potong(teks)
        print ("INI HASIL POTONG ",chunks)
        semua_chunk.extend(chunks)
        print (semua_chunk)
        print()
    # Proses Embeddings pake BATCH (Sekaligus banyak)
    database = []
    batch_size = 15 # Kirim 15 chunk sekali jalan
    total = len(semua_chunk)
    print (total)
    print()
    print (f"Memproses {total} chunk dengan sistem batching...")
    for i in range(0, total, batch_size):
        batch = semua_chunk[i : i + batch_size]
        print (batch)
        res_embeddings = get_embeddings(batch)

        if res_embeddings:
            for t, e in zip(batch, res_embeddings):
                database.append({"text": t, "embeddings": e})
            print (f"Progress: {min(i + batch_size, total)}/{total} selesai...")

    simpan(nama_db, database)
else:
    print ("Database ditemukan, langsung memuat...")

new = load_data(PATH_FILE)
print ("\nSEBELUM DI POTONG\n",new)
print ("Panjangnya adalah ",len(new))
print()



semua_chunk = []
for teks in new:
    chunks = potong(teks)
    semua_chunk.extend(chunks)
print ("\nSESUDAH DIPOTONG\n",semua_chunk)
print ("Panjangnya adalah ",len(semua_chunk))


def load_data(location):
    if location.endswith(".txt"):
        with open(location,"r")as f:
            lines =[line.strip() for line in f if line.strip()]
            print ("DATA TXT BERHASIL DIMUAT...")
            return lines
    elif location.endswith(".pdf"):
        with open(location,"rb") as f:
            reader = PdfReader(f)
            file = []
            for halaman in reader.pages:
                teks_halaman = halaman.extract_text()
                if teks_halaman:
                    file.append(teks_halaman)
            return file


