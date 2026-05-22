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

