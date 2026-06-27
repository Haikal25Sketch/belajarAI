import os
import sqlite3
import logging
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.sqlite import SqliteSaver
from tools import DAFTAR_TOOLS
from logging_config import setup_logging
from config import MODEL

# 1. Setup Logging & Env
load_dotenv()
logger = setup_logging()

# 2. Inisialisasi Model & Tools
model = ChatGroq(model=MODEL, temperature=0.7)

# 3. Setup Memori (Checkpointer)
# Gunakan sqlite3.connect dan pastikan check_same_thread=False
DB_PATH = "Lilim_memories.db"
conn = sqlite3.connect(DB_PATH, check_same_thread=False) #Thread disini tuh adalah pekerja bukan seperti thread_id
memori= SqliteSaver(conn)

# 4. Definisi System Prompt (Dibuat lebih ketat agar tool calling lancar)
SYSTEM_PROMPT ="""Kamu adalah AI Agent bernama LILIM.
Kamu memiliki tool dan bisa digunakan jika diperlukan,jika
tidak ,gunakan pengetahuan umum kamu,berbincanglah denganku seakan aku adalah teman dekatmu."""

# 5. Buat Graph (Agent)
# create_react_agent otomatis mengurus looping dan tool calling
app = create_react_agent(
    model, 
    tools=DAFTAR_TOOLS, 
    checkpointer=memori,
    prompt=SYSTEM_PROMPT
)

# 6. Fungsi Utama untuk Memanggil Agent
def agent(pertanyaan):
    # thread_id adalah kunci memori di LangGraph
    config = {"configurable": {"thread_id": "Haikal_session"}}
    input_data = {"messages": [("user", pertanyaan)]}

    try:
        # Jalankan Graph
        hasil = app.invoke(input_data, config)
        
        
        # Ambil jawaban terakhir dari AI
        jawaban_ai = hasil["messages"][-1].content # jawaban ada di paling bawah
        
        print(f"\nLilim: {jawaban_ai}")
        
    except Exception as e:
        logger.error(f"Error pada LangGraph Agent: {e}")
        print(f"\nLilim: Maaf Haikal, ada kendala teknis... ({str(e)})")

if __name__ == "__main__":
    while True:
        tanya = input("\nKamu: ")
        if tanya.lower() == "keluar":
            break
        agent(tanya)

# ================================================================================
# PENJELASAN ELEMEN-ELEMEN BARU (LANGGRAPH):
# ================================================================================
# 1. create_react_agent:
#    Fungsi siap pakai (prebuilt) dari LangGraph untuk membuat ReAct Agent (Reasoning & Acting).
#    Fungsi ini secara otomatis menyusun struktur Graph yang terdiri dari node LLM dan node Tools,
#    serta mengelola loop/siklus pemanggilan tool secara otomatis tanpa perlu kita definisikan manual.
#
# 2. SqliteSaver & checkpointer:
#    SqliteSaver adalah salah satu checkpointer bawaan LangGraph yang menggunakan SQLite untuk 
#    menyimpan status percakapan (State) ke dalam database secara permanen. Ini membuat agen 
#    memiliki memori jangka panjang dan pendek (persisten) bahkan setelah script di-restart.
#
# 3. thread_id (di dalam config):
#    Pengenal unik untuk sesi percakapan saat ini. LangGraph menggunakan thread_id untuk memisahkan
#    dan mengambil memori yang sesuai untuk user tertentu. Jika thread_id sama, memori akan terus 
#    berlanjut; jika thread_id berbeda, percakapan akan dianggap sebagai sesi baru.
#
# 4. app.invoke(input_data, config):
#    Metode untuk memicu/menjalankan alur kerja graph. Kita mengirimkan input awal (messages) 
#    dan konfigurasi (seperti thread_id). app.invoke akan mengeksekusi graf hingga selesai, 
#    dan mengembalikan state akhir yang berisi daftar seluruh pesan percakapan (riwayat).

