import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser

# ==========================================
# MATERI BELAJAR LANGCHAIN DASAR (LCEL)
# ==========================================
# Referensi: Implementasi ChatGroq + PromptTemplate
# ==========================================

# 1. SETUP LINGKUNGAN
# Load API Key dari file .env (GROQ_API_KEY)
load_dotenv()

"""
KOMPONEN UTAMA LANGCHAIN YANG DIGUNAKAN:
1. ChatPromptTemplate: Mengatur struktur pesan (System, Human, AI).
2. ChatGroq: Interface untuk berinteraksi dengan model LLM dari Groq.
3. StrOutputParser: Menyederhanakan output model menjadi teks string.
4. LCEL (LangChain Expression Language): Menggunakan simbol pipe (|) untuk merantai komponen.
"""

# 2. PROMPT TEMPLATE (Template Perintah)
# Memungkinkan pembuatan prompt yang dinamis menggunakan placeholder {variable}.
# Keuntungan: Memisahkan logika instruksi dari data input.
prompt = ChatPromptTemplate.from_messages([
    ("system", "Kamu adalah {asisten} yang ahli dan selalu menjawab dengan sedikit cara dan langsung ke intinya."),
    ("human", "Jelaskan konsep {topik} dengan bahasa yang mudah dimengerti.")
])

# 3. CHAT MODEL (Otak AI)
# Menggunakan Groq sebagai penyedia model. 
# Parameter:
# - model: Nama model yang digunakan (misal: llama-3.3-70b-versatile).
# - temperature: Mengatur kreativitas (0.0 = kaku/faktual, 1.0 = sangat kreatif).
model = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.7
)

# 4. OUTPUT PARSER (Pembersih Jawaban)
# Secara default, LLM mengembalikan objek kompleks (AIMessage). 
# StrOutputParser mengambil teks kontennya saja agar mudah ditampilkan.
parser = StrOutputParser()

# 5. CHAINING / PERANTAIAN (Inti dari LangChain)
# Menggunakan syntax LCEL: input -> prompt -> model -> parser -> output
# Ini membuat kode lebih modular dan mudah dibaca.
chain = prompt | model | parser

# 6. EKSEKUSI (Menjalankan Program)
def jalankan_tutorial():
    print("--- MEMULAI SESI BELAJAR LANGCHAIN ---")
    print("Status: Memanggil API Groq via LangChain...")

    # Invoke: Menjalankan chain dengan input berupa dictionary
    try:
        hasil = chain.invoke({
            "asisten": "Guru matematika",
            "topik": "rumus kuantum"
        })
        # invoke : cara standar memasukkan input ke chain
        print("\n=== JAWABAN DARI AI ===")
        print(hasil)
        print("\n========================")
        print("Pelajaran Selesai: Kamu baru saja belajar cara merantai (chain) Prompt, Model, dan Parser!")
    
    except Exception as e:
        print(f"Terjadi kesalahan: {e}")
        print("Pastikan GROQ_API_KEY sudah terpasang di .env")

if __name__ == "__main__":
    jalankan_tutorial()

#MENAMBAHKAN INGATAN(CHAT HISTORY)

# IMPORT MODUL INGATAN BAWAAN LANGCHAIN
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

load_dotenv()

# ==========================================
# 1. PROMPT TEMPLATE DENGAN PLACEHOLDER HISTORY
# ==========================================
prompt = ChatPromptTemplate.from_messages([
    ("system", "Kamu adalah Lilim, AI Assistant yang ramah dan asyik diajak ngobrol."),
    # MessagesPlaceholder adalah "slot kosong" otomatis tempat LangChain menyelipkan 
    # seluruh riwayat obrolan masa lalu secara rapi sebelum dikirim ke Groq
    MessagesPlaceholder(variable_name="riwayat_chat"),
    ("human", "{input}")
])

# ==========================================
# 2. CHAT MODEL & PARSER (Sama seperti kemarin)
# ==========================================
model = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7)
parser = StrOutputParser()

# Rangkaian dasar kita (Prompt -> Model -> Parser)
chain_dasar = prompt | model | parser

# ==========================================
# 3. MANAJEMEN INGATAN (DATABASE MEMORI)
# ==========================================
# Kita buat dictionary kosong untuk menampung ingatan di dalam RAM.
# Di dunia nyata, ini bisa diganti dengan database SQLite/Redis agar awet.
penyimpanan_memori = {}

def ambil_riwayat_sesi(session_id: str):
    """Fungsi pembantu untuk mengambil atau membuat sesi obrolan baru berdasarkan ID"""
    if session_id not in penyimpanan_memori:
        penyimpanan_memori[session_id] = ChatMessageHistory()
    return penyimpanan_memori[session_id]

# ==========================================
# 4. BUNGKUS CHAIN DASAR AGAR OTOMATIS BERIKATAN DENGAN MEMORI
# ==========================================
chain_dengan_memori = RunnableWithMessageHistory(
    chain_dasar,
    get_session_history=ambil_riwayat_sesi,
    input_messages_key="input", # Variabel input dari user
    history_messages_key="riwayat_chat" # Slot kosong di prompt template tadi
)

# ==========================================
# 5. LOOP INTERAKTIF CHAT (SEPERTI CHATGPT ASLI)
# ==========================================
if __name__ == "__main__":
    print("=== LILIM CHATBOT READY (Ketik 'keluar' untuk berhenti) ===")
    
    # ID Sesi unik, biar kalau ada user lain chat, ingatannya gak ketukar
    konfigurasi = {"configurable": {"session_id": "sesi_koding_kamu"}}
    
    while True:
        user_input = input("\nKamu: ")
        if user_input.lower() == "keluar":
            print("Lilim: Bye-bye!")
            break
            
        if not user_input.strip():
            continue
            
        # Panggil chain dengan mengirim input + ID Konfigurasi Sesi
        print("Lilim berpikir...")
        jawaban = chain_dengan_memori.invoke(
            {"input": user_input},
            config=konfigurasi
        )
        
        print(f"Lilim: {jawaban}")

