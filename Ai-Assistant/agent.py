import os
import logging
import warnings
from dotenv import load_dotenv
from logging_config import setup_logging
from config import MODEL
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
logger = setup_logging()
#IMPORT MODUL MODUL LANGCHAIN
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
#IMPORT TOOL 
from tools import DAFTAR_TOOLS

SYSTEM_PROMPT = """Kamu adalah AI Agent bernama LILIM.
Kamu memiliki tool dan bisa digunakan jika diperlukan,jika tidak ,gunakan pengetahuan umum kamu,namaku Haikal."""
#DAFTARKAN PROMPT
prompt = ChatPromptTemplate.from_messages([
    ("system",SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="riwayat_chat"),
    ("human", "{input}"),
    # SLOT WAJIB UNTUK AGENT: Tempat LangChain mengelola logika pengerjaan alat
    MessagesPlaceholder(variable_name="agent_scratchpad") # Gaboleh diganti variabelnya
])
#MASUKKAN MODEL
model = ChatGroq(model=MODEL,temperature=0.7)

#BUAT OTAK AI
lilim_assistant = create_openai_tools_agent(model,DAFTAR_TOOLS,prompt)
#BUAT PENGEKSEKUSI/TUBUH AI
eksekutor_agent= AgentExecutor(agent=lilim_assistant,tools=DAFTAR_TOOLS,verbose=False)

#KONEKSIKAN DENGAN SQLITE
URL_DATABASE = "sqlite:///Lilim_memories.db"

def ambil_riwayat_sqlite(session_id: str):
    return SQLChatMessageHistory(
        session_id=session_id,
        connection=URL_DATABASE,
        table_name="chat_history"
    )

#BUNGKUS AGENT DENGAN MEMORI
agent_dengan_memori = RunnableWithMessageHistory(
    eksekutor_agent,
    get_session_history=ambil_riwayat_sqlite,
    input_messages_key="input",
    history_messages_key="riwayat_chat"
    )

#KONFIGURASI UNTUK SESSION ID PERCAKAPAN
konfigurasi = {"configurable": {"session_id": "Haikal_session"}}
#PERSINGKAT DEF AGENT
def agent(pertanyaan):
    try:
        jawaban = agent_dengan_memori.invoke(
        {"input":pertanyaan},
        config =konfigurasi
        )
       # Menampilkan jawaban AI (jawaban aslinya ada di key 'output')
        print(f"\nLilim: {jawaban['output']}")
    except Exception as e:
        logger.error(f"Error pada Agent: {e}")
        print(f"\nLilim: Maaf, terjadi kesalahan sistem... ({str(e)})")
