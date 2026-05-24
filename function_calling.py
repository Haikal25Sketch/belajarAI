import requests
import os
import json
from dotenv import load_dotenv
import math
import logging
from PyPDF2 import PdfReader
import sqlite3
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



PATH_FILE = "/storage/emulated/0/Download/IDN BROKEN STRINGS.pdf"
#Mengganti tempat menampung database dari json menjadi sqlite
NAMA_DB_SQLITE = "Database_IDN BROKEN STRINGS.db"

#Inisialisasi Database Sqlite
koneksi = sqlite3.connect(NAMA_DB_SQLITE) # Membuka koneksi ke file database SQLite. Jika file database dengan nama tersebut belum ada, SQLite akan otomatis membuat file baru di direktori tersebut. Dibuat di awal script
cursor = koneksi.cursor() # Membuat objek Cursor. Analoginya, jika connect adalah membuka gerbang bank data, maka cursor adalah teller/petugas yang bertugas mondar-mandir mengeksekusi perintahkamu ke dalam ruangan brankas.Dibuat setelah koneksi berhasil

#MEMBUAT TABEL (Teks dan embeddings disimpan dalam TEKT/BLOB)
cursor.execute("""
CREATE TABLE IF NOT EXISTS dokumen (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    text TEXT,
    embeddings TEXT
)
""")

koneksi.commit() # MENYIMPAN SEMUA PERUBAHAN KE DATABASE

#INTEGER : TIPE DATANYA ANGKA BULAT
#PRIMARY KEY : PENANDA BAHWA KOLOM INI ADALAH IDENTITAS TIAP BARIS
#AUTOINCREMENT : MEMKASA SQLITE AGAR TIDAK MEMAKAI ID LAMA / NILAI id OTOMATIS BERTAMBAH SENDIRI

#CEK APAKAH DATABASE KOSONG

cursor.execute("SELECT COUNT(*) FROM dokumen") # Menghitung jumlah baris di tabel dokumen
if cursor.fetchone()[0] == 0:
    logger.info("Membuat database baru...")
    data_mentah = load_data(PATH_FILE)

    # 1. Kumpulkan semua chunk dulu
    semua_chunk = []
    for teks in data_mentah:
        chunks = potong(teks)
        semua_chunk.extend(chunks)

    # Proses Embeddings pake BATCH (Sekaligus banyak)
    BATCH_SIZE = 15
    batch_size = BATCH_SIZE # Kirim 15 chunk sekali jalan
    total = len(semua_chunk)

    logger.info(f"Memproses {total} chunk dengan sistem batching ke SQlite...")
    for i in range(0, total, batch_size):
        batch = semua_chunk[i : i + batch_size]
        res_embeddings = get_embeddings(batch)

        if res_embeddings:
            for t, e in zip(batch, res_embeddings):
            #Ubah list embeddings jadi string json agar bisa disimpan di teks SQlite
                embedding_str = json.dumps(e)
                cursor.execute(
                    "INSERT INTO dokumen(text,embeddings) VALUES(?,?)",(t,embedding_str)
                )
                
            logger.info(f"Progress SQlite: {min(i + batch_size, total)}/{total} selesai...")


else:
    logger.info("Database SQlite ditemukan, langsung memuat...")

TOOLS_DEFINITION = [
    {
        "type": "function",
        "function": {
            "name": "cuaca",
            "description": "Cek cuaca kota",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Nama kota"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function", 
        "function": {
            "name": "kalkulator",
            "description": "Hitung matematika",
            "parameters": {
                "type": "object",
                "properties": {
                    "ekspresi": {"type": "string", "description": "Ekspresi matematika"}
                },
                "required": ["ekspresi"]
            }
        }
    },
    {
        "type":"function",
        "function": {
            "name":"cari_database",
            "description":"Cari info di database/PDF",
            "parameters": {
                "type":"object",
                "properties": {
                    "query": {"type":"string", "description":"Kata kunci"}
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
    logger.info(f"Mencari info di Database untuk query {query}")
    embs = get_embeddings([query])
    if not embs :
        logger.info("Gagal memproses query ke Embedding")
    user_emb= embs[0]
    # Ambil semua data dari SQlite ke memori secara sekaligus
    cursor.execute("SELECT text,embeddings FROM dokumen")
    baris_data = cursor.fetchall() # Mengambil semua baris
    hasil = []
    for teks,emb_str in baris_data:
        #kembalikan string json jadi list float python
        emb_list = json.loads(emb_str)
        #bandinglan
        skor = banding(emb_list,user_emb)
        hasil.append({"skor":skor,"text":teks})

    BATAS_AKURASI = 0.4
    hasil.sort(key=lambda x:x["skor"],reverse = True)
    top_search =[h for h in hasil if h["skor"] > BATAS_AKURASI] [:5]
    return "\n".join([h["text"] for h in top_search])



TOOLS = {
    "cuaca": cuaca,
    "kalkulator": kalkulator,
    "cari_database":cari_database
}
#MEMPERBARUI PROMPT
SYSTEM_PROMPT = """Kamu adalah AI Agent yang memiliki nama Lilim. Tugasmu adalah membantu hal yang diminta user tapi dengan metode step by step untuk melatih pemikiran.

Kamu juga memiliki beberapa tool:
1.cuaca
2.kalkulator
3.cari_database

Ini adalah contoh perilaku user agar kamu memahami polanya:

1.user:"Bagaimana cuaca...."
maka kamu bisa menggunakan tool cuaca

2.user:"Berapa hasil dari 2 * 4"
maka kamu bisa menggunakan tool kalkulator

3.user:"siapa nama orang dalam database yang aku kirim"
maka kamu bisa menggunakan tool cari_database

Jika jawaban dari pertanyaan user tidak ada dalam tool, maka kamu bisa menggunakan pengetahuan umum kamu untuk memberikan jawabannya."""
# ===== 3. AGENT =====
messages = [{"role":"system","content":SYSTEM_PROMPT}]

def agent(pertanyaan):
    groq_token = os.getenv("GROQ_API_KEY")
    headers = {
        "Authorization": f"Bearer {groq_token}",
        "Content-Type": "application/json"
    }
    
    # Batasi history agar tidak meledak (Simpan system prompt + 9 pesan terakhir)
    if len(messages) > 10:
        messages[:] = [messages[0]] + messages[-9:]
        
    messages.append({"role":"user","content":pertanyaan})

    while True:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json={
                "model": "llama-3.3-70b-versatile",
                "messages": messages,
                "tools": TOOLS_DEFINITION
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
            argumen = json.loads(tool_call["function"]["arguments"]) # Ubah teks json menjadi string python

            #print(f"[Tool dipanggil: {nama_tool} | Input: {argumen}]")
            # Jalankan fungsi Python
            hasil = TOOLS[nama_tool](**argumen)
            #print(f"[Hasil: {hasil}]")
            
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
try:
    while True:
        user = input("\nKamu: ")
        if user.lower() == "keluar":
            break
        agent(user)
finally:
    logger.info("Menutup database SQLite...")
    cursor.close()
    koneksi.close()
