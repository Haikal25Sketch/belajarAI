"""
1. Apa itu Database? (Analogi Lemari Arsip vs Catatan Berantakan)
​Bayangkan kamu punya toko sepatu.
​File Biasa (.txt atau .csv): Ini seperti kamu mencatat semua penjualan di selembar kertas atau buku tulis biasa. Kalau catatanmu masih sedikit, tidak masalah. 

Tapi bayangkan kalau catatanmu sudah ribuan halaman, lalu kamu mau mencari "Sepatu ukuran 42 warna merah yang terjual di bulan Januari". Kamu harus membalik halaman satu per satu. Repot dan lama banget, kan?
​
DATABASE: Ini seperti lemari arsip digital yang super rapi dan punya asisten pintar. Semua data disimpan dalam kotak-kotak khusus (disebut Tabel). Kalau kamu butuh data, kamu tinggal bilang ke si asisten, "Tolong ambilkan data sepatu ukuran 42 warna merah bulan Januari", dan boom! Si asisten langsung mengambilkannya dalam kedipan mata.
​
2. Kenapa Pakai SQLite3 di Python?
​Di dunia nyata, database itu biasanya besar dan butuh komputer server sendiri (seperti MySQL atau PostgreSQL). Ribet buat kita yang baru belajar.
​
Nah, SQLite itu ibarat "database kantong". Dia sangat ringan, tidak perlu instalasi yang aneh-aneh, dan langsung jadi satu file biasa di komputermu. 
Python sudah menyediakan SQLite secara bawaan, jadi kita tinggalpakai! Cocok banget buat belajar karena simpel tapi kekuatannya sama dengan database besar.
​
3. Tipe Data di SQLite (Bahasa Planetnya SQLite)
​Sebelum membuat tabel, kita harus tahu jenis data apa saja yang bisa disimpan. 
Di SQLite, bahasanya sangat sederhana dibanding database lain. Cukup ingat 4 ini dulu:
​
TEXT: Untuk teks/tulisan (Contoh: Nama orang, judul buku, alamat).
​
INTEGER: Untuk angka bulat (Contoh: Umur, jumlah barang, tahun).
​REAL: Untuk angka pecahan/desimal (Contoh: Harga barang, tinggi badan, IPK).
​
NULL: Kalau datanya kosong atau belum diisi.
​
4. Praktek: Membuat Koneksi dan Tabel Pertama
​Sekarang, mari kita buat database dan tabel pertama kita menggunakan Python. Kita akan membuat database untuk Perpustakaan Mini.
​Ibaratnya, kita akan:
​Membeli lemari arsipnya (Membuat file database).
​Membuat laci khusus untuk buku (Membuat Tabel bernama buku).
"""

import sqlite3

# ==========================================
# 5. METODE-METODE LENGKAP SQLITE3 DI PYTHON
# ==========================================

# --- A. KONEKSI & CURSOR ---

# 1. sqlite3.connect("nama_file.db")
# Membuka koneksi ke file database. Jika file tidak ada, otomatis dibuatkan yang baru.
koneksi_istri = sqlite3.connect("istri.db")
koneksi_perpus = sqlite3.connect("perpustakaan.db")

# 2. koneksi.cursor()
# Membuat objek cursor. Cursor bertugas mengeksekusi perintah SQL dan mengambil hasil data.
cursor_istri = koneksi_istri.cursor()
cursor_perpus = koneksi_perpus.cursor()

# --- B. EKSEKUSI PERINTAH (SQL EXECUTION) ---

# 3. cursor.execute("PERINTAH SQL")
cursor_istri.execute("""
CREATE TABLE IF NOT EXISTS informasi(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    nama TEXT,
    asal TEXT,
    umur INTEGER
)
""")

# 4. cursor.executemany("PERINTAH SQL", LIST_DATA)
# TIPS: Kita cek dulu apakah tabel kosong, supaya tidak duplikat setiap kali file ini dijalankan
cursor_istri.execute("SELECT COUNT(*) FROM informasi")
daftar_informasi =[
        ("HuTao","Liyue",18),
        ("YaeMiko","Inazuma",20),
        ("Hoshino","Kivotos",18)
    ]
if cursor_istri.fetchone()[0] == 0:
    cursor_istri.executemany("INSERT INTO informasi(nama,asal,umur) VALUES(?,?,?)", daftar_informasi)

# 5. cursor.executescript("BEBERAPA PERINTAH SQL")
cursor_istri.executescript("""
     CREATE TABLE IF NOT EXISTS kategori (id INTEGER, nama TEXT);
     INSERT OR IGNORE INTO kategori VALUES (1, 'Fiksi');
 """)

# --- C. MENGAMBIL DATA (FETCHING) ---

# Jalankan SELECT dulu sebelum fetch
cursor_istri.execute("SELECT * FROM informasi")
cursor_perpus.execute("SELECT * FROM buku") # Pastikan tabel 'buku' sudah ada dari latihan sebelumnya

# 6. cursor.fetchone()
data_istri = cursor_istri.fetchone()
data_perpus = cursor_perpus.fetchone()

print (f"Fetchone Istri\nNama:{data_istri[1]}\nAsal:{data_istri[2]}\nUmur:{data_istri[3]}")
print()
print(f"Fetchone Buku\nJudul:{data_perpus[1]}\nPenulis:{data_perpus[2]}\nTahun terbit:{data_perpus[3]}")
print()
# 7. cursor.fetchmany(n)
data_beberapa_istri = cursor_istri.fetchmany(2)
data_beberapa_perpus= cursor_perpus.fetchmany(2)
print (f"Fetchmany(2) Istri\nNama:{data_beberapa_istri[0][1]}\nAsal:{data_beberapa_istri[0][2]}\nUmur:{data_beberapa_istri[0][3]}")
print (f"Fetchmany(2) Istri\nNama:{data_beberapa_istri[1][1]}\nAsal:{data_beberapa_istri[1][2]}\nUmur:{data_beberapa_istri[1][3]}")

print()
print (f"Fetchmany(2) Buku\nJudul:{data_beberapa_perpus[0][1]}\nPenulis:{data_beberapa_perpus[0][2]}\nTahun terbit:{data_beberapa_perpus[0][3]}")
print (f"Fetchmany(2) Buku\nJudul:{data_beberapa_perpus[1][1]}\nPenulis:{data_beberapa_perpus[1][2]}\nTahun terbit:{data_beberapa_perpus[1][3]}")

# 8. cursor.fetchall()
# RESET: Isi ulang cursor
cursor_istri.execute("SELECT * FROM informasi") 
cursor_perpus.execute("SELECT * FROM buku") 

data_semua_istri = cursor_istri.fetchall()
data_semua_buku = cursor_perpus.fetchall()

print(f"Semua Istri: {data_semua_istri}")
print(f"Semua Buku: {data_semua_buku}")

# --- D. PENYIMPANAN & PENUTUPAN ---

koneksi_istri.commit()
koneksi_perpus.commit()

cursor_istri.close()
cursor_perpus.close()
koneksi_istri.close()
koneksi_perpus.close()

# --- TIPS TAMBAHAN: MENGGUNAKAN WITH (CONTEXT MANAGER) ---
with sqlite3.connect("istri.db") as conn:
    curr = conn.cursor()
    curr.execute("SELECT * FROM informasi")
    print (f"\nSemua data (via with) = {curr.fetchall()}")

# ==========================================
# 6. MANIPULASI DATA LANJUTAN (WHERE, UPDATE, DELETE, JOIN)
# ==========================================
print()

db = "toko_buah.db"
connect_toko = sqlite3.connect(db)
cursor_toko = connect_toko.cursor()
cursor_toko.execute("""
    CREATE TABLE IF NOT EXISTS rak(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    price INTEGER,
    stock INTEGER
    )
""")
daftar_buah = [
    ("Jeruk",5000,20),
    ("Kiwi",6000,10),
    ("Leci",3000,8)
]
cursor_toko.execute("SELECT COUNT(*) FROM rak")
if cursor_toko.fetchone()[0] == 0:
    cursor_toko.executemany("INSERT INTO rak(name,price,stock) VALUES(?,?,?)",daftar_buah)
connect_toko.commit()

cursor_toko.execute("SELECT name,stock FROM rak")

for buah in cursor_toko.fetchall():
    print (f"BUAH:{buah[0]} | STOK:{buah[1]}")
print()
print("===WHERE===") #FILTER DATA 

cursor_toko.execute("SELECT name,stock FROM rak WHERE stock < 15")
for bit in cursor_toko.fetchall():
    print (f"BUAH SEDIKIT:{bit[0]} | STOK:{bit[1]}")
print()

# UPDATE DATA
print ("===UPDATE===")
cursor_toko.execute("UPDATE rak SET price=7000 WHERE name ='Jeruk'")
print("DATA JERUK TELAH DIUPDATE...")
connect_toko.commit()
print()
print("===DELETE===") # HAPUS DATA
cursor_toko.execute("DELETE FROM rak WHERE name ='Kiwi'")
print("DATA KIWI TELAH DIHAPUS...")
connect_toko.commit()
print()
cursor_toko.execute("SELECT * FROM rak")
print ("===DATA BUAH TOKO LILIM===")
for buah in cursor_toko.fetchall():
    print (f"NO.{buah[0]}|BUAH:{buah[1]} | HARGA:{buah[2]}| STOCK:{buah[3]}")
print()
#MENGGABUNGKAN TABEL
print ("===JOIN===")
cursor_toko.execute("""
CREATE TABLE IF NOT EXISTS transaksi(
    id_transaksi INTEGER PRIMARY KEY AUTOINCREMENT,
    buah_id INTEGER,
    jumlah_terjual INTEGER
    )
""")
# MEMASUKAN DATA TRANSAKSI
data_penjualan=[
    (1,5), #1 adalah jeruk 5 adalah yang terjual
    (3,7),
    (1,2),
    (3,4)
]
cursor_toko.execute("SELECT COUNT(*) FROM transaksi")
if cursor_toko.fetchone()[0] == 0:
    cursor_toko.executemany("INSERT INTO transaksi(buah_id,jumlah_terjual) VALUES(?,?)",data_penjualan)
    connect_toko.commit()
print("TABEL TRANSAKSI TELAH DIBUAT...")
#PENGGABUNGAN DENGAN INNER JOIN
cursor_toko.execute("""
    SELECT rak.name,transaksi.jumlah_terjual 
    FROM rak
    INNER JOIN transaksi ON transaksi.buah_id = rak.id
""") # DUA TABEL DIHUBUNGKAN OLEH ID ,DI RAK NAMANYA rak.id DI TRANSAKSI NAMANYA transaksi.buah_id

connect_toko.commit()
print()
print ("===LAPORAN PENJUALAN TOKO LILIM===")
for hasil in cursor_toko.fetchall():
    print (hasil)
    print (f"BUAH:{hasil[0]} | TERJUAL:{hasil[1]} biji")
    connect_toko.commit()

# EKSEKUSI DATA AGREGASI (SUM & GROUP BY)
print()

cursor_toko.execute("""
    SELECT rak.name, SUM(transaksi.jumlah_terjual)
    FROM transaksi 
    INNER JOIN rak ON transaksi.buah_id = rak.id
    GROUP BY rak.name
""")

print("=== TOTAL BUAH YANG TERJUAL ===")
for hasil in cursor_toko.fetchall():
    print(f"Buah: {hasil[0]} | Total Terjual: {hasil[1]} biji")

connect_toko.close()

# ORDER BY & LIMIT

# ORDER BY : DIPAKAI UNTUK MENGURUTKAN DATA BERDASARKAN KOLOM TERTENTU,
# 2 PENGURUTAN ORDER BY:
# 1. ASC (Ascending) A-Z / 0-10
# 2. DESC (Descending) Z-A /10-0

# LIMIT :DIPAKAI UNTUK MEMOTONG HASIL DATA,JADI WALAUPUN HASILNYA BANYAK YANG DIPAKAI YA SESUAI AMA LIMIT YANG LU TENTUKAN
connect_toko = sqlite3.connect("toko_buah.db")
cursor_toko = connect_toko.cursor()
cursor_toko.execute("SELECT rak.name,rak.price FROM rak ORDER BY price ASC LIMIT 1")

print ("===TERMURAH - TERMAHAL===")
for price in cursor_toko.fetchall():
    print (f"BUAH: {price[0]} | HARGA : {price[1]} ")
