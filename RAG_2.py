import requests
import os
from dotenv import load_dotenv
import math
import json
import logging

"""Membuat RAG yang bisa membaca dari file"""

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

#Memgambil data dari file lain
def load_data(location):
    with open(location,"r")as f:
        lines =[line.strip() for line in f if line.strip()]
        logger.info("DATA BERHASIL DIMUAT...")
        return lines

#Simpan hasil ke json
def simpan(location,data):
    with open(location,"w") as f:
        json.dump(data,f,indent=4)
        logger.info(f"DATA BERHASIL DISIMPAN DI {f}")
def ambil(location):
    with open(location,"r") as f:
        file = json.load(f)
        return file
def potong(teks,ukuran=3,overlap=2):
    kata = teks.split()
    hasil = []

    langkah = ukuran-overlap #(ukuran dan overlap jngn sama)
    for i in range(0,len(kata),langkah):
        potongan = kata[i:i+ukuran]
        hasil.append(" ".join(potongan))

        if i+ukuran >= len(kata):
            break

    return hasil

#Set-up API
token = os.getenv("HUGGINGFACE_TOKEN")

url = "https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L6-v2/pipeline/feature-extraction"


headers ={
    "Authorization":f"Bearer {token}",
    "Content-Type":"application/json"
    }

# Set -up Untuk mendapatkan embeddings
def get_embeddings(text):
    token = os.getenv("HUGGINGFACE_TOKEN")

    url = "https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L6-v2/pipeline/feature-extraction"


    headers ={
        "Authorization":f"Bearer {token}",
        "Content-Type":"application/json"
        }

    payload = {"inputs":text}
    try:
        response = requests.post(url,headers=headers,json=payload)
        if response.status_code == 200:
            logger.info("BERHASIL MENDAPATKAN EMBEDDINGS...")
            return response.json()
        else:
            logger.error(f"EMBEDDINGS GAGAL DIDAPATKAN | PENYEBAB : {response.status_code}")
        return None
    except requests.exceptions.ConnectionError:
        logger.error("KONEKSI INTERNET MATI,TIDAK BISA MENGAKSES!")
    except requests.exceptions.Timeout:
        logger.warning("TERLALU LAMA!!")

    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTPS ERROR!!! : {e} ")

    except requests.exceptions.RequestException as e :
        logger.error(f"ERROR!!! {e} ")

# Bandingkan embeddings
def banding(a,b):
    dot = sum(x*y for x,y in zip(a,b))
    mag_a = math.sqrt(sum(x**2 for x in a))
    mag_b = math.sqrt(sum(x**2 for x in b))

    if mag_a == 0 or mag_b == 0:
        return 0

    return dot / (mag_a * mag_b)


#Cari yang terbaik
def input_user():
    user_input = input("Masukkan kalimat : ")
    return user_input



#EKSEKUSI

load_dotenv()
data = load_data("pengetahuan.txt")


#Cek keberadaan database,kalo ada gaperlu request ke Huggingface lagi
if os.path.exists("Database_RAG.json"):
    data_awal = ambil("Database_RAG.json")
    if len(data) != len(data_awal): #Antisipasi jika ada perubahan pada database
        database = []

        for text in data:
            chunking = potong(text)
            for chunk in chunking:
                database.append({
                "text":chunk,
                "embeddings":get_embeddings([chunk])[0]
                })
            simpan("Database_RAG.json",database)

    else:
        data_awal = ambil("Database_RAG.json")

else:
    database = []

    for text in data:
        chunking = potong(text)
        for chunk in chunking:
            database.append({
            "text":chunk,
            "embeddings":get_embeddings([chunk])[0]
                })

#Simpan dan akses data json
    simpan("Database_RAG.json",database)
data_awal = ambil("Database_RAG.json")

# --- LOOP PERTANYAAN BERULANG ---
while True:
    user = input("\nMasukkan pertanyaan (ketik 'keluar' untuk berhenti): ")

    if user.lower() in ['keluar', 'exit', 'quit']:
        print("Terima kasih! Sampai jumpa.")
        break

    if not user.strip():
        continue

    # Ambil embedding untuk pertanyaan baru
    res_embeddings = get_embeddings([user])
    if not res_embeddings:
        continue
    user_embedding = res_embeddings[0]

    #Cari yang terbaik,termirip dan simpan hasilnya
    hasil_pencarian = []

    for emb in data_awal:
        vec = emb["embeddings"]
        teks = emb["text"]
        skor = banding(vec, user_embedding)
        hasil_pencarian.append({"skor": skor, "text": teks})

    # Diurutkan berdasarkan SKOR (bukan teks) dari besar ke kecil
    hasil_pencarian.sort(key=lambda x: x["skor"], reverse=True)

    # Ambil 5 terbaik
    BATAS_AKURASI = 0.5
    top_search = [h for h in hasil_pencarian if h["skor"] > BATAS_AKURASI][:5]

    # Gabungkan teksnya
    context_gabungan = "\n".join([item["text"] for item in top_search])

    groq_token = os.getenv("GROQ_API_KEY")

    groq_headers = {
            "Authorization": f"Bearer {groq_token}",
            "Content-Type": "application/json"
    }

    groq_data = {
            "model": "llama-3.3-70b-versatile",
            "messages": [
            {"role":"system",
            "content":"Kamu adalah asisten pribadi yang akan menjawab pertanyaan berdasarkan data yang saya berikan,jika yang aku tanyakan tidak ada dalam data yang saya berikan,jawab sesuai dengan pengetahuan umummu"},
            {"role":"user",
            "content":f"KONTEKS:\n{context_gabungan}\n\nPERTANYAAN: {user}"
            }
            ]
    }

    try:
        response = requests.post("https://api.groq.com/openai/v1/chat/completions",
        headers=groq_headers,json=groq_data)

        if response.status_code == 200:
            hasil =response.json()
            jawaban_ai = hasil["choices"][0]["message"]["content"]
            print ("\n===JAWABAN AI===")
            print (jawaban_ai)
        else:
            print(f"Error ke Groq: {response.status_code} - {response.text}")

    except Exception as e:
        print(f"Gagal menghubungi Groq: {e}")

