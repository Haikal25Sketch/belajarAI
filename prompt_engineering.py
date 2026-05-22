# ==========================================
# MATERI: PROMPT ENGINEERING DASAR
# ==========================================
# Prompt Engineering adalah teknik memberikan instruksi yang tepat 
# agar AI memberikan hasil yang sesuai keinginan kita.

"""
1. SYSTEM PROMPT YANG EFEKTIF
------------------------------
System Prompt adalah "identitas" atau "aturan main" yang kita berikan di awal.
Tujuan: Mengatur gaya bahasa, batasan, dan peran AI.

CONTOH IMPLEMENTASI:
messages = [
    {
        "role": "system", 
        "content": "Kamu adalah asisten ahli Python yang ketat. Selalu jawab dengan kode dan penjelasan singkat. Jangan gunakan bahasa gaul."
    },
    {"role": "user", "content": "Apa itu list?"}
]
"""

"""
2. FEW-SHOT PROMPTING
---------------------
Teknik memberikan satu atau lebih contoh (shot) agar AI memahami pola yang kita mau.
Sangat efektif untuk tugas klasifikasi atau format jawaban tertentu.

CONTOH IMPLEMENTASI:
prompt = \"\"\"
Klasifikasikan sentimen dari teks berikut.
Contoh:
Input: "Saya sangat suka belajar Python!"
Output: POSITIF

Input: "Kodenya error terus, saya pusing."
Output: NEGATIF

Input: "Hari ini saya belajar coding."
Output: NETRAL

Input: \"{user_input}\"
Output: \"\"\"
\"\"\"

"""
"""
3. CHAIN OF THOUGHT (CoT)
-------------------------
Menyuruh AI untuk berpikir secara bertahap (step-by-step) sebelum memberikan jawaban akhir.
Tujuan: Meningkatkan akurasi untuk soal logika atau perhitungan rumit.

CONTOH IMPLEMENTASI:
prompt = \"\"\"
Selesaikan masalah matematika ini dengan berpikir langkah demi langkah:
"Jika Budi punya 5 apel, lalu membeli 2 keranjang yang masing-masing berisi 10 apel, berapa total apel Budi?"

Mari kita hitung:
1. Hitung jumlah apel di keranjang baru...
2. Tambahkan dengan apel awal...
Jawaban akhir: ...
\"\"\"
"""

# Praktek Langsung dalam format Agent Sederhana
import os

def simple_agent(user_query):
    # Gabungan 3 Teknik:
    # 1. System Prompt (Roleplay)
    # 2. Few-shot (Memberikan pola jawaban)
    # 3. Chain of Thought (Minta AI berpikir bertahap)
    
    system_instruction = (
        "Kamu adalah Agent Logika. Ikuti aturan ini:\n"
        "1. Selalu awali jawaban dengan 'Analisis:' untuk berpikir step-by-step (Chain of Thought).\n"
        "2. Berikan jawaban akhir di baris baru dengan format 'HASIL AKHIR: ...'.\n"
        "3. Contoh pola:\n"
        "User: Berapa 2+2*5?\n"
        "Analisis: Perkalian dilakukan lebih dulu. 2*5 = 10. Lalu 2 + 10 = 12.\n"
        "HASIL AKHIR: 12"
    )
    
    print("--- PROMPT YANG DIKIRIM ---")
    print(f"System: {system_instruction}")
    print(f"User: {user_query}")
    print("---------------------------\n")

if __name__ == "__main__":
    query = "Jika saya punya 3 kotak, masing-masing isi 4 coklat, dan saya makan 2 coklat, sisa berapa?"
    simple_agent(query)
    print("Petunjuk: Jalankan file ini untuk melihat bagaimana prompt disusun!")
