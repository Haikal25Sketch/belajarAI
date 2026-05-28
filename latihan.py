import sqlite3

db = "toko_buah.db"
connect = sqlite3.connect(db)
cursor = connect.cursor()
cursor.execute("""
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
cursor.execute("SELECT COUNT(*) FROM rak")
if cursor.fetchone()[0] == 0:
    cursor.executemany("INSERT INTO rak(name,price,stock) VALUES(?,?,?)",daftar_buah)
connect.commit()

cursor.execute("SELECT name,stock FROM rak")

for buah in cursor.fetchall():
    print (f"BUAH:{buah[0]} | STOK:{buah[1]}")
print()
cursor.execute("SELECT name,stock FROM rak WHERE stock < 15")
for bit in cursor.fetchall():
    print (f"BUAH SEDIKIT:{bit[0]} | STOK:{bit[1]}")

# UPDATE DATA
cursor.execute("UPDATE rak SET price=7000 WHERE name ='Jeruk'")
connect.commit()
cursor.execute("DELETE FROM rak WHERE name ='Kiwi'")
connect.commit()
print()
cursor.execute("SELECT * FROM rak")
print ("===DATA BUAH TOKO LILIM===")
for buah in cursor.fetchall():
    print (f"NO.{buah[0]}|BUAH:{buah[1]} | HARGA:{buah[2]} | STOCK:{buah[3]}")
