import requests
import math
import json
import os
import logging
from dotenv import load_dotenv
from config import THRESHOLD,TOP_K,NAMA_DB,PATH_FILE,BATCH_SIZE
from PyPDF2 import PdfReader
load_dotenv()
logger = logging.getLogger(__name__)
token = os.getenv("HUGGINGFACE_TOKEN")
url = "https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L6-v2/pipeline/feature-extraction"
hf_headers = {
    "Authorization":f"Bearer {token}",
    "Content-Type":"application/json"
}
def load_data(location):
    if not os.path.exists(location):
        logger.error(f"FILE {location} TIDAK ADA!!!.")
        return []
    try:
        if location.endswith(".txt"):
            with open(location,"r")as f:
                lines =[line.strip() for line in f if line.strip()]
                logger.info("DATA TXT BERHASIL DIMUAT...")
                return lines

        elif location.endswith(".pdf"):
            try:
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

            except (FileNotFoundError,PermissionError,UnicodeDecodeError) as e:
                logger.error(f"ERROR PDF : {e}!!!")
                return []
    except (FileNotFoundError,PermissionError) as e:
        logger.error(f"ERROR FILE : {e}!!!")
        return []

    logger.error("FORMAT FILE TIDAK DIKENALI!!!")
    raise ValueError("FORMAT FILE TIDAK DIDUKUNG")


def simpan(location,data):
    with open(location,"w") as f:
        json.dump(data,f,indent=4)
        logger.info(f"DATA BERHASIL DISIMPAN DI {location}")

def ambil(location):
    with open(location,"r") as f:
        file = json.load(f)
        return file

# Potong teks jadi kecil-kecil
def potong(teks, ukuran=500, overlap=100): # Ukuran default digedein biar AI lebih pinter
    kata = teks.split() # pecah string jadi list
    hasil = []
    langkah = ukuran - overlap
    for i in range(0, len(kata), langkah):
        potongan = kata[i:i+ukuran]
        hasil.append(" ".join(potongan))
        if i + ukuran >= len(kata):
            break
    return hasil

def get_embeddings(text):
    payload = {"inputs": text}
    try:
        response = requests.post(url, headers=hf_headers, json=payload)
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"EMBEDDINGS GAGAL | STATUS: {response.status_code}")
            return None
    except Exception as e:
        logger.error(f"ERROR API: {e}")
        return None

 
def banding(a, b):
    dot = sum(x*y for x,y in zip(a,b))
    mag_a = math.sqrt(sum(x**2 for x in a))
    mag_b = math.sqrt(sum(x**2 for x in b))
    if mag_a == 0 or mag_b == 0: return 0
    return dot / (mag_a * mag_b)

def cari_database(query):
    data_awal = ambil(NAMA_DB)
    data = load_data(PATH_FILE)

    semua_chunk = []
    for teks in data:
        chunks = potong(teks)
        semua_chunk.extend(chunks)
    if len(data_awal) != len(semua_chunk):
        logger.info("DATA BERUBAH!!! RE-EMBED...")
        database = []
        batch_size = BATCH_SIZE
        total = len(semua_chunk)

        logger.info(f"Memproses {total} chunk dengan sistem batching...")

        for i in range(0, total, batch_size):
            batch = semua_chunk[i : i + batch_size]
            res_embeddings = get_embeddings(batch)

            if res_embeddings:
                for t, e in zip(batch, res_embeddings):
                    database.append({"text": t, "embeddings": e})
            logger.info(f"Progress: {min(i + batch_size, total)}/{total} selesai...")
        simpan(NAMA_DB,database)
        data_awal = database

    else:
        logger.info("DATABASE SESUAI. LANGSUNG PAKAI...")

    embs = get_embeddings([query])
    user_emb= embs[0]

    hasil = []
    for emb in data_awal:
        skor = banding(emb["embeddings"],user_emb)
        hasil.append({"skor":skor,"text":emb["text"]})

    BATAS_AKURASI = THRESHOLD
    hasil.sort(key=lambda x:x["skor"],reverse = True)
    top_search =[h for h in hasil if h["skor"] > BATAS_AKURASI] [:TOP_K]
    return "\n".join([h["text"] for h in top_search])

data = load_data("/storage/emulated/0/Download/8294.pdf")

