import os
import warnings
from typing import TypedDict, Annotated, Sequence
from dotenv import load_dotenv

# Membungkam peringatan Deprecation agar terminal tetap bersih
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Load environment variables (seperti GROQ_API_KEY)
load_dotenv()

"""
================================================================================
          PANDUAN BELAJAR LANGGRAPH UNTUK CALON AI ENGINEER
================================================================================

1. APA ITU LANGGRAPH?
   LangGraph adalah library dari LangChain untuk membangun aplikasi berbasis multi-agent
   dan agentic workflow yang memiliki siklus/looping (cycles).
   
   Perbedaan Utama:
   - LangChain LCEL (Rantai Biasa): Bersifat linier/searah (A -> B -> C). Tidak bisa kembali ke belakang.
   - LangGraph (Graf): Bersifat siklus (A -> B -> A -> C). Agent bisa mencoba suatu tindakan, mengevaluasi 
     hasilnya, dan jika salah atau kurang lengkap, agent bisa mengulangi tindakan tersebut sampai benar.

2. MENGAPA SEORANG AI ENGINEER PERLU LANGGRAPH?
   Di industri AI modern, sistem Agentic AI tidak lagi hanya "sekali tanya, sekali jawab".
   Agent harus bisa:
   - Mengambil keputusan sendiri secara iteratif.
   - Mengoreksi kodenya sendiri jika error (Self-correction).
   - Menggunakan berbagai Tools secara berulang.
   - Mempertahankan memori percakapan multi-sesi (Persistence).

3. ARSITEKTUR UTAMA LANGGRAPH:
   a. STATE (Keadaan/Status):
      Satu-satunya sumber kebenaran (source of truth) berupa struktur data (biasanya Python TypedDict atau Pydantic)
      yang diakses dan diperbarui oleh setiap Node di dalam graf.
   b. NODES (Simpul/Titik):
      Fungsi Python biasa yang menerima 'State' saat ini, melakukan proses/logika (misal: panggil LLM, run tool),
      lalu mengembalikan data baru untuk memperbarui 'State'.
   c. EDGES (Sisi/Garis Hubung):
      Aturan perpindahan dari satu node ke node lainnya. Ada dua jenis:
      - Normal Edge: Selalu berpindah dari Node A ke Node B.
      - Conditional Edge: Memutuskan ke mana arah alur berikutnya berdasarkan isi State (menggunakan fungsi logika).
   d. CHECKPOINTER (Penyimpan Memori):
      Sistem database (seperti SqliteSaver) untuk menyimpan snapshot State secara otomatis di setiap langkah.

4. APA ITU TYPEDDICT DAN STATEGRAPH? (PEMANGGILAN METHODNYA)
   a. TypedDict (dari modul 'typing'):
      - Pengertian: TypedDict digunakan untuk memberikan petunjuk tipe data (type hints) pada dictionary.
        Ini mendefinisikan key apa saja yang wajib/boleh ada beserta tipe datanya (misal: query: str).
      - Runtime Behavior (Saat Dijalankan): TypedDict HANYA ada sebagai type hint. Di runtime, object ini
        adalah dictionary Python biasa (`dict`).
      - Pengaksesan Method: Karena ia adalah dictionary biasa, ia TIDAK memiliki method `.add()`. Anda akan
        menemui error `AttributeError` jika memanggil `.add()`. Anda hanya bisa mengakses datanya menggunakan
        tanda kurung siku (misal: `state["query"]`) atau method dictionary biasa seperti `.get()`, `.update()`, dll.
      - Jika Menjadi Parameter Fungsi/Class:
        Contoh: `def node_analis(state: AgentState):`
        Di dalam fungsi ini, parameter `state` adalah dictionary. Anda tidak bisa menggunakan `state.add(...)`
        atau `state.add_node(...)`. Anda hanya menggunakannya untuk membaca/menulis data state.

   b. StateGraph (dari modul 'langgraph.graph'):
      - Pengertian: StateGraph adalah class pembangun utama dari LangGraph. Objek dari class inilah yang kita
        gunakan untuk merancang alur graf (menambahkan simpul, menghubungkan garis, dll).
      - Runtime Behavior (Saat Dijalankan): Ia adalah object instance dari class StateGraph.
      - Pengaksesan Method: Ya! Objek StateGraph MEMILIKI method bawaan untuk merakit graf, seperti:
        * `.add_node(nama_node, fungsi_node)` -> untuk mendaftarkan node baru.
        * `.add_edge(node_asal, node_tujuan)` -> untuk menghubungkan dua node secara langsung.
        * `.add_conditional_edges(...)` -> untuk menghubungkan node dengan percabangan bersyarat.
        * `.compile()` -> untuk mengompilasi/merakit struktur graf menjadi aplikasi siap pakai.
        Catatan: Ia tidak memiliki method `.add()` polos, melainkan method-method spesifik di atas.
      - Jika Menjadi Parameter Fungsi/Class:
        Contoh: `def setup_workflow(workflow: StateGraph):`
        Di dalam fungsi/class tersebut, karena parameter `workflow` adalah instance StateGraph, Anda BISA
        mengakses dan memanggil method seperti `workflow.add_node(...)` atau `workflow.add_edge(...)`.

================================================================================
"""

# Kita akan mencoba membuat CUSTOM GRAPH sederhana untuk mensimulasikan Agen AI
# tanpa perlu bergantung pada API Key agar Haikal bisa mencobanya langsung di Termux.
# Namun, kami juga menyediakan kode integrasi LLM di bagian bawah.

# ================================================================================
# TAHAP 1: MENDAPATKAN KATEGORI UTAMA & IMPORT DARI LANGGRAPH
# ================================================================================
from langgraph.graph import StateGraph, START, END

# ================================================================================
# TAHAP 2: MENENTUKAN STATE (Bentuk Data Graf)
# ================================================================================
# State ini menyimpan data yang akan dialirkan antar node.
class AgentState(TypedDict):
    query: str          # Pertanyaan pengguna
    analisis: str       # Hasil analisis sementara dari Agent
    sumber_data: str    # Dari mana data diambil (Database / Pengetahuan Umum)
    jawaban: str        # Jawaban akhir untuk pengguna

# ================================================================================
# TAHAP 3: MENULIS NODES (Logika Komputasi)
# ================================================================================
# Setiap node menerima state saat ini dan mengembalikan dictionary yang akan memperbarui state.

def node_analis(state: AgentState):
    """Menganalisis pertanyaan untuk menentukan jalur pencarian terbaik."""
    print("[Node Analis] Menganalisis pertanyaan...")
    pertanyaan = state["query"].lower()
    
    # Aturan sederhana: jika ada kata 'gudang' atau 'stok', arahkan ke database gudang
    if "gudang" in pertanyaan or "stok" in pertanyaan or "barang" in pertanyaan:
        sumber = "database_gudang"
    else:
        sumber = "pengetahuan_umum"
        
    return {
        "analisis": f"Pertanyaan diidentifikasi membutuhkan informasi dari: {sumber}",
        "sumber_data": sumber
    }

def node_database(state: AgentState):
    """Node yang mensimulasikan pencarian di Database Gudang."""
    print("[Node Database] Mengakses database gudang...")
    # Simulasi data gudang
    return {
        "jawaban": "Hasil Query DB: Stok beras di gudang saat ini adalah 250 ton dan minyak goreng 120 liter."
    }

def node_umum(state: AgentState):
    """Node yang mensimulasikan jawaban dari Pengetahuan Umum LLM."""
    print("[Node Umum] Mengakses pengetahuan umum...")
    return {
        "jawaban": "Menurut pengetahuan umum saya: Indonesia merupakan negara kepulauan terbesar di dunia dengan ribuan pulau."
    }

def node_penyusun_jawaban(state: AgentState):
    """Node akhir untuk merapikan format jawaban."""
    print("[Node Penyusun] Merapikan jawaban akhir...")
    jawaban_rapi = f"Halo Haikal!\n{state['jawaban']}\n(Sumber: {state['sumber_data']})"
    return {
        "jawaban": jawaban_rapi
    }

# ================================================================================
# TAHAP 4: MENULIS LOGIKA EDGES BERSYARAT (Conditional Router)
# ================================================================================
# Fungsi ini menentukan node mana berikutnya yang harus dikunjungi.
def router_sumber_data(state: AgentState):
    print(f"[Router] Mengecek analisis sumber data: {state['sumber_data']}")
    if state["sumber_data"] == "database_gudang":
        return "ke_database"
    else:
        return "ke_umum"

# ================================================================================
# TAHAP 5: MENYUSUN GRAF (Membangun Struktur)
# ================================================================================
# 1. Inisialisasi builder StateGraph dengan State yang sudah dibuat
workflow = StateGraph(AgentState)

# 2. Tambahkan Nodes ke dalam workflow
workflow.add_node("analis", node_analis)
workflow.add_node("database", node_database)
workflow.add_node("umum", node_umum)
workflow.add_node("penyusun", node_penyusun_jawaban)

# 3. Hubungkan alur dengan Edges
# Alur awal: Masuk (START) langsung menuju Node Analis
workflow.add_edge(START, "analis")

# Alur bersyarat: Dari Node Analis, arahkan berdasarkan hasil router
workflow.add_conditional_edges(
    "analis", 
    router_sumber_data,
    {
        "ke_database": "database",
        "ke_umum": "umum"
    }
)

# Alur akhir: Setelah selesai dari database atau umum, arahkan ke penyusun
workflow.add_edge("database", "penyusun")
workflow.add_edge("umum", "penyusun")

# Setelah dari penyusun, selesaikan perjalanan graf (END)
workflow.add_edge("penyusun", END)

# ================================================================================
# TAHAP 6: COMPILE & RUN
# ================================================================================
# Compile mengubah struktur blueprint menjadi aplikasi executable yang siap dijalankan
app = workflow.compile()

def jalankan_langgraph_simulasi():
    print("\n" + "="*50)
    print("DEMO 1: SIMULASI LANGGRAPH KUSTOM (TANPA API KEY)")
    print("="*50)
    
    # Percobaan 1: Tanya stok gudang (seharusnya masuk ke Node Database)
    pertanyaan_1 = {"query": "Berapa stok beras di gudang kita?"}
    print(f"\nUser: {pertanyaan_1['query']}")
    hasil_1 = app.invoke(pertanyaan_1)
    print(f"Jawaban Akhir:\n{hasil_1['jawaban']}")
    
    print("\n" + "-"*30)
    
    # Percobaan 2: Tanya umum (seharusnya masuk ke Node Umum)
    pertanyaan_2 = {"query": "Berapa jumlah pulau di Indonesia?"}
    print(f"\nUser: {pertanyaan_2['query']}")
    hasil_2 = app.invoke(pertanyaan_2)
    print(f"Jawaban Akhir:\n{hasil_2['jawaban']}")

# ================================================================================
# TAHAP 7: CONTOH NYATA DENGAN MODEL LLM & MEMORI (INTEGRASI GROQ)
# ================================================================================
# Di bawah ini adalah referensi kode bagaimana cara membuat agen dengan LLM asli
# yang memiliki memori percakapan persisten.

def contoh_kode_llm_riil():
    """
    KODE INI BISA DIJALANKAN JIKA GROQ_API_KEY SUDAH DI-SET DI FILE .env
    Sama seperti implementasi LILIM di 'agent_graph.py'"""

    from langchain_groq import ChatGroq
    from langgraph.prebuilt import create_react_agent
    from langgraph.checkpoint.sqlite import SqliteSaver
    import sqlite3
    
    # 1. Inisialisasi LLM
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7)
    
    # 2. Buat database memori SQLite
    conn = sqlite3.connect("memori_percakapan.db", check_same_thread=False)
    memory = SqliteSaver(conn)
    
    # 3. Definisikan tools yang bisa dipakai LLM
    def hitung_pajak(pendapatan: int) -> float:
        '''Fungsi menghitung pajak 10% dari pendapatan'''
        return pendapatan * 0.1
        
    daftar_tools = [hitung_pajak]
    
    # 4. Buat React Agent bawaan LangGraph
    # create_react_agent secara otomatis menyusun Graf ReAct:
    # LLM -> Tool? -> Ya (Jalankan Tool) -> Panggil LLM lagi -> Selesai (Kembalikan Jawaban)
    agent_app = create_react_agent(
        llm, 
        tools=daftar_tools, 
        checkpointer=memory,
        prompt="Kamu adalah AI Asisten Pajak yang ramah."
    )
    
    # 5. Jalankan dengan thread_id agar memori tersimpan
    config = {"configurable": {"thread_id": "sesi_haikal_01"}}
    
    # Chat Pertama
    respons1 = agent_app.invoke(
        {"messages": [("user", "Nama saya Haikal dan pendapatan saya 10.000. Berapa pajaknya?")]},
        config
    )
    print(respons1["messages"][-1].content)
    
    # Chat Kedua (LLM akan ingat nama 'Haikal' karena thread_id-nya sama)
    respons2 = agent_app.invoke(
        {"messages": [("user", "Siapa nama saya tadi dan berapa hasil pajaknya?")]},
        config
    )
    print(respons2["messages"][-1].content)


    pass

if __name__ == "__main__":
    jalankan_langgraph_simulasi()
    print("\n" + "="*50)
    print("TIPS AI ENGINEER:")
    print("1. Kuasai konsep 'State'. State adalah jantung dari LangGraph.")
    print("2. Pelajari kapan harus membuat StateGraph kustom (seperti contoh di atas)")
    print("   dan kapan harus menggunakan prebuilt agent (seperti `create_react_agent`).")
    print("3. Coba jalankan script ini dengan mengetik: python belajarAI/belajar_Langgraph.py")
    print("="*50)
