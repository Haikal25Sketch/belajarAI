import requests
import os
from dotenv import load_dotenv
import math
import json
import logging
from PyPDF2 import PdfReader

"""Membuat RAG yang bisa membaca dari file (Versi Optimasi PDF)"""

def setup_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    terminal_handler = logging.StreamHandler()
    terminal_handler.setLevel(logging.DEBUG)

    stream_fmt = logging.Formatter("%(levelname)s |  %(message)s")
    terminal_handler.setFormatter(stream_fmt)

    logger.addHandler(terminal_handler)
    return logger

logger = setup_logging()

# Memgambil data dari file lain
def load_data(location):
    if location.endswith(".txt"):
        with open(location,"r")as f:
            lines =[line.strip() for line in f if line.strip()]
            logger.info("DATA TXT BERHASIL DIMUAT...")
            return lines
    elif location.endswith(".pdf"):
        with open(location,"rb") as f:
            reader = PdfReader(f)
            file = []
            for halaman in reader.pages:
                teks_halaman = halaman.extract_text()
                if teks_halaman:
                    # Simpan per halaman, jangan kumulatif biar RAM gak penuh
                    file.append(teks_halaman) 
            logger.info(f"DATA PDF BERHASIL DIMUAT ({len(file)} Halaman)...")
            return file
    
# Simpan hasil ke json
def simpan(location,data):
    with open(location,"w") as f:
        json.dump(data,f,indent=4)
        logger.info(f"DATA BERHASIL DISIMPAN DI {location}")

def ambil(location):
    with open(location,"r") as f:
        file = json.load(f)
        return file

# Potong teks jadi kecil-kecil
def potong(teks, ukuran=100, overlap=20): # Ukuran default digedein biar AI lebih pinter
    kata = teks.split()
    hasil = []
    langkah = ukuran - overlap 
    for i in range(0, len(kata), langkah):
        potongan = kata[i:i+ukuran]
        hasil.append(" ".join(potongan))
        if i + ukuran >= len(kata):
            break
    return hasil

# Set-up API
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
        response = requests.post(url, headers=headers, json=payload)
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"EMBEDDINGS GAGAL | STATUS: {response.status_code}")
            return None
    except Exception as e:
        logger.error(f"ERROR API: {e}")
        return None

# Bandingkan kemiripan (Cosine Similarity)
def banding(a, b):
    dot = sum(x*y for x,y in zip(a,b))
    mag_a = math.sqrt(sum(x**2 for x in a))
    mag_b = math.sqrt(sum(x**2 for x in b))
    if mag_a == 0 or mag_b == 0: return 0
    return dot / (mag_a * mag_b)

# --- EKSEKUSI ---
# Ganti path sesuai file yang mau dibaca
path_file = "pengetahuan.txt"
# Ambil nama file buat jadi nama database biar gak ketuker sama file lain
nama_file_asli = os.path.basename(path_file).split('.')[0]
nama_db = f"Database_{nama_file_asli}.json"

# Cek keberadaan database, kalo ada gaperlu request ke Huggingface lagi
if not os.path.exists(nama_db):
    logger.info("Membuat database baru...")
    data_mentah = load_data(path_file)
    
    # 1. Kumpulkan semua chunk dulu
    semua_chunk = []
    for teks in data_mentah:
        chunks = potong(teks)
        semua_chunk.extend(chunks)
    
    # 2. FIX POIN 2: Proses Embeddings pake BATCH (Sekaligus banyak)
    database = []
    batch_size = 15 # Kirim 15 chunk sekali jalan
    total = len(semua_chunk)
    
    logger.info(f"Memproses {total} chunk dengan sistem batching...")
    for i in range(0, total, batch_size):
        batch = semua_chunk[i : i + batch_size]
        res_embeddings = get_embeddings(batch)
        
        if res_embeddings:
            for t, e in zip(batch, res_embeddings):
                database.append({"text": t, "embeddings": e})
            logger.info(f"Progress: {min(i + batch_size, total)}/{total} selesai...")
    
    simpan(nama_db, database)
else:
    logger.info("Database ditemukan, langsung memuat...")

# Ambil data yang sudah jadi
data_awal = ambil(nama_db)

# --- LOOP PERTANYAAN ---
while True:
    user = input("\nMasukkan pertanyaan (ketik 'keluar' untuk berhenti): ")
    if user.lower() in ['keluar', 'exit', 'quit']: 
        print("Terima kasih! Sampai jumpa.")
        break
    if not user.strip(): continue

    # Ambil embedding buat pertanyaan user
    res = get_embeddings([user])
    if not res: continue
    user_embedding = res[0]

    # Cari yang paling mirip
    hasil_pencarian = []
    for emb in data_awal:
        skor = banding(emb["embeddings"], user_embedding)
        hasil_pencarian.append({"skor": skor, "text": emb["text"]})

    hasil_pencarian.sort(key=lambda x: x["skor"], reverse=True)
    
    # Ambil 5 teratas yang masuk akal
    BATAS_AKURASI = 0.4
    top_search = [h for h in hasil_pencarian if h["skor"] > BATAS_AKURASI][:5]
    context_gabungan = "\n".join([item["text"] for item in top_search])

    # Tanya ke Groq
    groq_token = os.getenv("GROQ_API_KEY")
    groq_headers = {"Authorization": f"Bearer {groq_token}", "Content-Type": "application/json"}
    groq_data = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {"role":"system", "content":"Kamu adalah asisten pribadi yang akan menjawab pertanyaan berdasarkan data yang saya berikan, jika yang aku tanyakan tidak ada dalam data yang saya berikan, jawab sesuai dengan pengetahuan umummu."},
            {"role":"user", "content":f"KONTEKS:\n{context_gabungan}\n\nPERTANYAAN: {user}"}
        ]
    }

    try:
        response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=groq_headers, json=groq_data)
        if response.status_code == 200:
            jawaban_ai = response.json()["choices"][0]["message"]["content"]
            print("\n=== JAWABAN AI ===")
            print(jawaban_ai)
        else:
            print(f"Error ke Groq: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Gagal menghubungi Groq: {e}")
