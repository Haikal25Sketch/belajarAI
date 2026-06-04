import os
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

load_dotenv()

# 1. Inisialisasi Pinecone
api_key = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=api_key)

NAMA_INDEX = "belajar-ai"

def setup_pinecone():
    # Cek apakah index sudah ada
    if NAMA_INDEX not in pc.list_indexes().names():
        print(f"Membuat index baru: {NAMA_INDEX}...")
        pc.create_index(
            name=NAMA_INDEX,
            dimension=384, # Sesuai dengan ukuran vektor 'all-MiniLM-L6-v2'
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1" # Region gratisan biasanya di sini
            )
        )
        print("Index berhasil dibuat!")
    else:
        print(f"Index '{NAMA_INDEX}' sudah siap digunakan.")

def simpan_ke_pinecone(id_teks, vektor, metadata):
    index = pc.Index(NAMA_INDEX)
    index.upsert(
        vectors=[
            {
                "id": id_teks, 
                "values": vektor, 
                "metadata": metadata
            }
        ]
    )
    print(f"Data '{id_teks}' berhasil disimpan ke Cloud!")

def cari_di_pinecone(vektor_query, top_k=3):
    index = pc.Index(NAMA_INDEX)
    hasil = index.query(
        vector=vektor_query,
        top_k=top_k,
        include_metadata=True
    )
    return hasil

if __name__ == "__main__":
    setup_pinecone()
