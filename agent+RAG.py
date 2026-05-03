import requests
import os
from dotenv import load_dotenv
import json
import math
load_dotenv()
 #Menambahkan RAG ke Agent Ai,dahal anjaay dahal dibantu ai
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





# ===== 1. TOOLS =====
def cuaca (query):
    database = {
    "jakarta":"10°C",
    "bogor":"13°C",
    "bekasi":"12°C"
    }
    query = query.lower()
    for key in database:
        if key in query:
            return database[key]
    return "Data tidak tersedia"


def kalkulator(ekspresi):
    try:
        return str(eval(ekspresi))
    except:
        return "Ekspresi tidak valid"

def cari_database(query):
    data_awal = ambil("Database_pengetahuan.json")
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


# ===== 2. PROMPT =====
SYSTEM_PROMPT = """Kamu adalah AI Agent.
Kamu punya tools:
- cuaca(query): cek cuaca di kota tertentu
- kalkulator(ekspresi): menghitung matematika
- cari_database(query): cari informasiku di database
Kalau butuh tool, balas HANYA dengan format:
TOOLS: nama_tool
INPUT: inputnya

Kalau sudah bisa jawab tanpa tool, langsung jawab saja."""

# ===== 3. AGENT LOOP =====
def agent(pertanyaan):
    groq_token = os.getenv("GROQ_API_KEY")
    headers = {
        "Authorization": f"Bearer {groq_token}",
        "Content-Type": "application/json"
    }

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": pertanyaan}
    ]

    while True:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json={"model": "llama-3.3-70b-versatile", "messages": messages}
        )


        balasan = response.json()["choices"][0]["message"]["content"]
        print (response.json())
        print(f"\nAI:\n{balasan}")

        if "TOOLS:" in balasan and "INPUT:" in balasan:
            baris = balasan.strip().split("\n")
            nama_tool = baris[0].replace("TOOLS:", "").strip()
            input_tool = baris[1].replace("INPUT:", "").strip()
            if nama_tool in TOOLS:
                hasil = TOOLS[nama_tool](input_tool)

                print(f"[Tool:{nama_tool} -> {hasil}]")

                # --- LANGKAH KRUSIAL: UPDATE MEMORI AGENT ---
                messages.append({"role": "assistant", "content": balasan})
                messages.append({"role": "user", "content": f"Hasil tool: {hasil}"})
            else:
                print(f"Tool {nama_tool} tidak ditemukan")
                break
        else:
            break

# ===== MAIN =====
while True:
    user = input("\nKamu: ")
    if user.lower() == "keluar":
        break
    agent(user)

"""
ALUR KERJA AI AGENT (STEP-BY-STEP):

1. INPUT USER: 
   Kamu memasukkan pertanyaan, misal: "Berapa suhu di Jakarta?".

2. INISIALISASI (MESSAGES):
   Program membuat daftar 'messages' yang berisi instruksi (System Prompt) 
   dan pertanyaanmu. Ini adalah "ingatan" si AI.

3. MIKIR (LLM REQUEST):
   Program mengirim 'messages' ke Groq (AI). 
   AI akan mikir: "Gue butuh tool cuaca buat jawab ini".

4. KEPUTUSAN (ACTION):
   AI membalas dengan format khusus: 'TOOLS: cuaca, INPUT: jakarta'. 
   Program mendeteksi tulisan "TOOLS:" tersebut.

5. EKSEKUSI (PYTHON WORK):
   Program (bukan AI) membedah teks tersebut, mengambil kata "cuaca" dan "jakarta", 
   lalu menjalankan fungsi python: cuaca("jakarta").

6. OBSERVASI (HASIL):
   Fungsi cuaca mengembalikan hasil: "10°C".

7. UPDATE INGATAN (APPEND) - [LANGKAH KRUSIAL]:
   Hasil "10°C" tadi dimasukkan kembali ke dalam daftar 'messages'. 
   Langkah ini sangat PENTING karena:
   - Mencegah Infinite Loop: Tanpa ini, AI akan lupa dia sudah memanggil tool dan akan terus-terusan memanggilnya lagi.
   - Memberi Konteks: Agar AI bisa membaca hasil tool dan memberikan jawaban akhir yang cerdas ke user berdasarkan data tersebut.

8. JAWABAN AKHIR:
   Program memanggil AI lagi (Loop kembali ke langkah 3). 
   Sekarang AI melihat di "ingatan"-nya sudah ada hasil tool "10°C". 
   AI langsung menjawab: "Suhu di Jakarta saat ini adalah 10°C".

9. SELESAI:
   Karena AI tidak minta tool lagi, program keluar dari Loop dan menunggumu bertanya lagi.
"""
