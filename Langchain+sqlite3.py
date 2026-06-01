import os
import warnings

# Bungkam SEMUA warning sebelum import langchain
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

import logging
logging.getLogger("langchain").setLevel(logging.ERROR)

from dotenv import load_dotenv
...
try:
    from langchain_core._api import LangChainDeprecationWarning
    warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)
except ImportError:
    pass
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
# 1. IMPORT MODUL SQLITE KHAS LANGCHAIN
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
# MENAMBAHKAN FUNCTION CALLING
# 1. IMPORT MODUL AGENT LANGCHAIN (Gunakan langchain_classic di lingkungan ini)
from langchain_classic.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.tools import tool
load_dotenv()


# MEMBUAT TOOLS (ALAT YANG BISA DIPAKAI AI)
@tool
def cuaca(query: str):
    """Check the weather in Jakarta, Bogor, or Bekasi."""
    database = {
        "jakarta": "10°C",
        "bogor": "13°C",
        "bekasi": "12°C"
    }
    for key in database:
        if key in query.lower():
            return database[key]
    return "Data tidak tersedia"

@tool
def kalkulator(ekspresi: str):
    """Evaluate a mathematical expression."""
    try:
        return str(eval(ekspresi))
    except Exception:
        return "Ekspresi tidak valid"

# MASUKKAN SEMUA TOOL KE LIST
daftar_tools = [cuaca, kalkulator]

# ==========================================
# 1. PROMPT TEMPLATE & CORE CHAIN
# ==========================================
prompt = ChatPromptTemplate.from_messages([
    ("system", "Kamu adalah Lilim, AI Assistant yang ramah. Ingatanmu disimpan permanen di SQLite."),
    MessagesPlaceholder(variable_name="riwayat_chat"),
    ("human", "{input}"),
    # SLOT WAJIB UNTUK AGENT: Tempat LangChain mengelola logika pengerjaan alat
    MessagesPlaceholder(variable_name="agent_scratchpad") # Gaboleh diganti variabelnya
])

model = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7)

# BERIKAN ALAT KE AI / RAKIT AGENT OTOMATIS
# lilim_assistant: Rencana/Instruksi (Otak)
lilim_assistant = create_openai_tools_agent(model, daftar_tools, prompt)
# eksekutor_agent: Tubuh/Manajer yang menjalankan instruksi & alat
eksekutor_agent = AgentExecutor(agent=lilim_assistant, tools=daftar_tools, verbose=True)

"""
PENGERTIAN GAMBLANG: Agent Executor
Agent Executor adalah "Mandor/Tubuh" si AI. 
AI Model cuma bisa "Berpikir", tapi Agent Executor lah yang benar-benar "Bertindak".

Alurnya:
1. AI bilang: "Saya butuh alat Kalkulator."
2. Agent Executor: Pergi ambil kalkulator, hitung, lalu kasih hasilnya ke AI.
3. AI: Merangkum hasil hitungan tadi buat kamu.

Tanpa Executor, AI cuma bisa berencana tanpa pernah bisa mengeksekusi alat (Tools).
"""

# ==========================================
# 2. KONEKSI SQLITE AUTOMATIC
# ==========================================
URL_DATABASE = "sqlite:///Lilim_memories.db"

def ambil_riwayat_sqlite(session_id: str):
    return SQLChatMessageHistory(
        session_id=session_id,
        connection=URL_DATABASE,
        table_name="chat_history"
    )

# ==========================================
# 3. BUNGKUS DENGAN MANAJER MEMORI SQLITE
# ==========================================
agent_dengan_memori = RunnableWithMessageHistory(
    eksekutor_agent,
    get_session_history=ambil_riwayat_sqlite,
    input_messages_key="input",
    history_messages_key="riwayat_chat"
)

# ==========================================
# PENGERTIAN ALUR: configurable -> get_session_history
# ==========================================
"""
Gimana ID Sesi dari invoke() bisa nyampe ke fungsi ambil_riwayat?
1. Kamu panggil invoke(..., config={"configurable": {"session_id": "Haikal"}})
2. LangChain melihat ada data di "configurable" dengan kunci "session_id".
3. LangChain otomatis ngoper nilai "Haikal" itu ke fungsi yang kamu daftarkan 
   di `get_session_history`, yaitu fungsi `ambil_riwayat_sqlite(session_id)`.
4. Fungsi itu lalu bikinin/ambilin database SQLite khusus buat si "Haikal".
"""

# ==========================================
# 4. LOOP INTERAKTIF CHAT
# ==========================================
if __name__ == "__main__":
    print("=== LILIM DENGAN PERMANENT SQLITE READY ===")
    konfigurasi = {"configurable": {"session_id": "Haikal_session"}}
    
    while True:
        user_input = input("\nKamu: ")
        if user_input.lower() == "keluar":
            print("Lilim: Bye-bye!")
            break
            
        if not user_input.strip():
            continue
            
        print("Lilim berpikir...")
        jawaban = agent_dengan_memori.invoke(
            {"input": user_input},
            config=konfigurasi
        )
        
        # Output Agent adalah dictionary, jawaban aslinya ada di key 'output'
        print(f"Lilim: {jawaban['output']}")
