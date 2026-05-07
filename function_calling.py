import requests
import os
import json
from dotenv import load_dotenv
import math
import logging
from PyPDF2 import PdfReader
load_dotenv()




def setup_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    terminal_handler = logging.StreamHandler()
    file_handler = logging.FileHandler("Agent_ai.log")
    terminal_handler.setLevel(logging.ERROR)
    file_handler.setLevel(logging.DEBUG)
    
    stream_fmt = logging.Formatter("%(levelname)s |  %(message)s")
    file_fmt =logging.Formatter("%(asctime)s | %(levelname)s|%(name)s | %(message)s")
    terminal_handler.setFormatter(stream_fmt)
    file_handler.setFormatter(file_fmt)
    logger.addHandler(terminal_handler)
    logger.addHandler(file_handler)
    return logger

logger = setup_logging()

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

# Fungsi dapetin embeddings (bisa terima satu teks atau list teks)
token = os.getenv("HUGGINGFACE_TOKEN")
url ="https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L6-v2/pipeline/feature-extraction"
hf_headers = {
    "Authorization":f"Bearer {token}",
    "Content-Type":"application/json"
}
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



PATH_FILE = "pengetahuan.txt"
NAMA_DB = "Database_pengetahuan.json"

if not os.path.exists(NAMA_DB):
    logger.info("Membuat database baru...")
    data_mentah = load_data(PATH_FILE)

    # 1. Kumpulkan semua chunk dulu
    semua_chunk = []
    for teks in data_mentah:
        chunks = potong(teks)
        semua_chunk.extend(chunks)

    # Proses Embeddings pake BATCH (Sekaligus banyak)
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

    simpan(NAMA_DB, database)
else:
    logger.info("Database ditemukan, langsung memuat...")

TOOLS_DEFINITION = [
    {
        "type": "function",
        "function": {
            "name": "cuaca",
            "description": "Cek cuaca di kota tertentu",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Gunakan jika user bertanya tentang kota atau suhu kota"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function", 
        "function": {
            "name": "kalkulator",
            "description": "Menghitung matematika",
            "parameters": {
                "type": "object",
                "properties": {
                    "ekspresi": {
                        "type": "string",
                        "description": "pertanyaan matematika yang harus dijawab"
                    }
                },
                "required": ["ekspresi"]
            }
        }
    },
    {
        "type":"function",
        "function": {
            "name":"cari_database",
            "description":"cari informasi tertentu di database",
            "parameters": {
                "type":"object",
                "properties": {
                    "query": {
                        "type":"string",
                        "description":"informasi yang harus dibandingkan di database"
                    }
                },
                "required":["query"]
             }
          }
       }
]


# Bandingkan kemiripan (Cosine Similarity)
def banding(a, b):
    dot = sum(x*y for x,y in zip(a,b))
    mag_a = math.sqrt(sum(x**2 for x in a))
    mag_b = math.sqrt(sum(x**2 for x in b))
    if mag_a == 0 or mag_b == 0: return 0
    return dot / (mag_a * mag_b)
# ===== 2. FUNGSI PYTHON BIASA =====
def cuaca(query):
    database = {
        "jakarta": "10°C",
        "bogor": "13°C",
        "bekasi": "12°C"
    }
    for key in database:
        if key in query.lower():
            return database[key]
    return "Data tidak tersedia"

def kalkulator(ekspresi):
    try:
        return str(eval(ekspresi))
    except:
        return "Ekspresi tidak valid"

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
        batch_size = 15
        total = len(semua_chunk)

        logger.info(f"Memproses {total} chunk dengan sistem batching...")
        
        for i in range(0, total, batch_size):
            batch = semua_chunk[i : i + batch_size]
            res_embeddings = get_embeddings(batch)

            if res_embeddings:
                for t, e in zip(batch, res_embeddings):
                    database.append({"text": t, "embeddings": e})
            logger.info(f"Progress: {min(i + batch_size, total)}/{total} selesai...")
        nama_file_asli = os.path.basename(PATH_FILE).split('.')[0]
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

    BATAS_AKURASI = 0.4
    hasil.sort(key=lambda x:x["skor"],reverse = True)
    top_search =[h for h in hasil if h["skor"] > BATAS_AKURASI] [:5]
    return "\n".join([h["text"] for h in top_search])



TOOLS = {
    "cuaca": cuaca,
    "kalkulator": kalkulator,
    "cari_database":cari_database
}

SYSTEM_PROMPT = """Kamu adalah AI Agent yang helpful. 
Gunakan tools yang tersedia jika diperlukan.
Jawab dalam Bahasa Indonesia."""

# ===== 3. AGENT =====
messages = [{"role":"system","content":SYSTEM_PROMPT}]

def agent(pertanyaan):
    groq_token = os.getenv("GROQ_API_KEY")
    headers = {
        "Authorization": f"Bearer {groq_token}",
        "Content-Type": "application/json"
    }
    messages.append({"role":"user","content":pertanyaan})

    while True:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json={
                "model": "llama-3.3-70b-versatile",
                "messages": messages,
                "tools": TOOLS_DEFINITION  # ← INI BEDANYA
            }
        )

        data = response.json()

        if "choices" not in data:
            logger.error(f"\n[ERROR API]: {data.get('error', {}).get('message', 'Terjadi kesalahan tidak dikenal')}")
            break # Keluar dari loop jika API error

        pesan = data["choices"][0]["message"]
        # Cek apakah AI minta jalankan tool
        if pesan.get("tool_calls"):
            tool_call = pesan["tool_calls"][0]
            
            nama_tool = tool_call["function"]["name"]
            argumen = json.loads(tool_call["function"]["arguments"])
            
            print(f"[Tool dipanggil: {nama_tool} | Input: {argumen}]")
            
            # Jalankan fungsi Python
            hasil = TOOLS[nama_tool](**argumen)
            print(f"[Hasil: {hasil}]")
            
            # Update memori
            messages.append(pesan)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "content": str(hasil)
            })

        else:
            # AI jawab langsung
            if pesan.get('content'):
                print(f"\nAI: {pesan['content']}")
            messages.append(pesan)
            break

# ===== MAIN =====
while True:
    user = input("\nKamu: ")
    if user.lower() == "keluar":
        break
    agent(user)
