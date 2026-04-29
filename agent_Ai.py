import requests
import os
from dotenv import load_dotenv

load_dotenv()

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

TOOLS = {
    "cuaca": cuaca,
    "kalkulator": kalkulator
}

# ===== 2. PROMPT =====
SYSTEM_PROMPT = """Kamu adalah AI Agent.
Kamu punya tools:
- cuaca(query): cek cuaca di kota tertentu
- kalkulator(ekspresi): menghitung matematika

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
       # print (response.json())

        balasan = response.json()["choices"][0]["message"]["content"]
        print (response.json())
        print(f"\nAI:\n {balasan}")

        if "TOOLS:" in balasan and "INPUT:" in balasan:
            baris = balasan.strip().split("\n")
            print (baris)
            nama_tool = baris[0].replace("TOOLS:", "").strip()
            input_tool = baris[1].replace("INPUT:", "").strip()

            if nama_tool in TOOLS:
                hasil = TOOLS[nama_tool](input_tool)
                print(f"[Tool {nama_tool} → {hasil}]")

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

7. UPDATE INGATAN (APPEND):
   Hasil "10°C" tadi dimasukkan kembali ke dalam daftar 'messages'. 
   Sekarang AI tahu kalau dia sudah cek cuaca dan hasilnya ada.

8. JAWABAN AKHIR:
   Program memanggil AI lagi (Loop kembali ke langkah 3). 
   Sekarang AI melihat di "ingatan"-nya sudah ada hasil tool "10°C". 
   AI langsung menjawab: "Suhu di Jakarta saat ini adalah 10°C".

9. SELESAI:
   Karena AI tidak minta tool lagi, program keluar dari Loop dan menunggumu bertanya lagi.
"""
