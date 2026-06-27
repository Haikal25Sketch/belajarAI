from fastapi import FastAPI
from pydantic import BaseModel
from agent_graph import app as agent
#inisialisasi loket API FastAPI
#FastAPI : cara agar Ai bisa diakses dari dunia luar
app_api=FastAPI(
    title="Lilim Ai-Assistant API",
    description="API untuk berinteraksi dengan Lilim"
    )
#Buat aturan untuk pengguna,ini kaya TypedDict tapi lebih keren
class ChatRequest(BaseModel): #Karena BaseModel dari pydantic yang dirancang untuk membaca JSON ,Memaksakan penggun untuk mengirimkan pesan dalam format JSON
    message : str
    thread_id : str ="default_session" # seperti config
    user_name : str
#Buat aturan jawaban yang dikirim Ai ke pengguna
class ChatResponse(BaseModel):
    response : str # ini jawaban akhir dari AI

#Membuat Endpoint utama(/chat) yang berfungsi untuk:
#1.Menerima ChatRequest
#2.Memasukkan pesan user dan Thread_id ke konfigurasi Langgraph
#3.Menjalankan Agen,menunggu berpikir,dan mengambil jawabanakhirnya
#4.Membungkus jawaban ke ChatResponse dan mengembalikannya ke user

@app_api.post("/chat",response_model=ChatResponse,status_code=200,tags=["Riwayat Haikal & Lilim"],summary="Nem7bak Chat ke lilim",description="My Endpoint",response_description="Jawaban Lilim")
#.post artinya loket ini menerima setoran data dari luar untuk diproses (bukan cuma sekadar minta data kosong).

#ini namanya Path Operation Configuration,bisa diisi:
#1.status_code:menentukan kode HTTP sukses,contoh
#@app_api.post("/chat",response_model=ChatResponse,status_code=200)

#2.tags:buat dokumentasi Swagger UI rapi,contoh
#@app_api.post("/chat",response_model=ChatResponse,tags=["ChatLilim"])

#3.summary & description:ngasih judul dan deskripsi,contoh
#@app_api.post("/chat",response_model=ChatResponse,summary:"Nembak chat ke Lilim",description="Endpoint utama untuk berinteraksi)

#4.dependencies: satpam global berfungsi agar hanya bisa diakses oleh yang punya token/API key,contoh
#@app_api.post("/chat",response_model=ChatResponse,dependencies=[Depends(daftarin function disini)])

#5.response_description: deskripsi output
#@app_api.post("/chat",response_model=ChatResponse,response_description="Jawaban Lilim")



def chat_endpoint(request:ChatRequest):
    #siapkan konfigurasi thread_id untuk memori Langgraph
    config = {"configurable":{"thread_id":request.thread_id,"user_name":request.user_name}}
    #masukkan pesan ke input graph
    inputs = {"messages":[("system",f"User memiliki nama {request.user_name} sapa dia diawal dengan namanya"),("user",request.message)]}
    #jalankan agent dengan sinkronus(.invoke())
    try:
        response_state=agent.invoke(inputs,config=config)
        jawaban=response_state["messages"][-1]
        jawaban_ai=jawaban.content
        #bungkus dan masukkan ke chatresponse
        return ChatResponse(response=jawaban_ai)

    except Exception as e:
        return ChatResponse(response=f"Error tidak terduga terjadi di agent : {str(e)} ")
        
#Endpoint untuk mengecek apakah API hidup atau tidak
@app_api.get("/")
def read():
    return f"Lilim is online"

# ================================================================================
# PENJELASAN PERINTAH CURL & UVICORN
# ================================================================================
#
# 1. PERINTAH CURL:
#    curl -X POST "http://127.0.0.1:8000/chat" \
#         -H "Content-Type: application/json" \
#         -d '{"message": "Halo Lilim, kenalan dong!", "thread_id": "kal_ganteng"}'
#
#    Penjelasan parameter-parameternya:
#    - "-X" (atau --request):
#      Menentukan metode HTTP (HTTP Method) yang digunakan untuk berinteraksi dengan server.
#      * Harus diisi apa: Diisi dengan metode HTTP seperti POST, GET, PUT, atau DELETE.
#      * Di perintah ini: Diisi dengan "POST" karena kita ingin mengirim/mengirimkan data baru ke endpoint "/chat".
#
#    - "-H" (atau --header):
#      Digunakan untuk mengirimkan metadata/informasi tambahan (HTTP Header) ke server.
#      * Harus diisi apa: Diisi dengan pasangan key-value header yang diperlukan oleh server.
#      * Di perintah ini: Diisi dengan "Content-Type: application/json" untuk memberi tahu server bahwa data yang dikirim di dalam body berformat JSON.
#
#    - "-d" (atau --data):
#      Digunakan untuk mengirimkan data payload/body request ke server.
#      * Harus diisi apa: Diisi dengan data yang ingin dikirimkan (dalam format JSON sesuai skema ChatRequest).
#      * Di perintah ini: Diisi dengan '{"message": "...", "thread_id": "..."}' yang berisi pesan obrolan dan ID sesi percakapan.
#
# 2. PERINTAH UVICORN:
#    uvicorn api:app_api --reload --host 0.0.0.0 --port 8000
#
#    Penjelasan komponen-komponennya:
#    - "uvicorn":
#      Program server web ASGI (Asynchronous Server Gateway Interface) berkinerja tinggi untuk Python yang digunakan untuk menjalankan aplikasi FastAPI agar bisa diakses.
#
#    - "api:app_api":
#      Menunjukkan lokasi objek aplikasi FastAPI yang ingin dijalankan.
#      * Harus diisi apa: Berformat "[nama_file_python_tanpa_ekstensi]:[nama_variabel_FastAPI]".
#      * Di perintah ini: "api" merujuk ke file "api.py" dan "app_api" merujuk ke variabel "app_api = FastAPI(...)" di baris 6.
#
#    - "--reload":
#      Flag/opsi untuk mengaktifkan fitur auto-reload.
#      * Maksudnya: Server akan mendeteksi setiap perubahan kode pada file python secara otomatis dan me-restart server sendiri secara instan tanpa perlu Anda matikan dan jalankan ulang secara manual. Sangat berguna saat tahap development (pengembangan).
#
#    - "--host 0.0.0.0":
#      Menentukan host/alamat IP tempat server akan berjalan dan mendengarkan request masuk.
#      * Harus diisi apa: Alamat IP host.
#      * Maksudnya: Nilai "0.0.0.0" berarti server dapat diakses dari semua kartu jaringan (network interfaces) perangkat, baik dari localhost (127.0.0.1) maupun dari luar perangkat lewat alamat IP jaringan lokal (Wi-Fi).
#
#    - "--port 8000":
#      Menentukan port jaringan yang digunakan untuk menjalankan aplikasi.
#      * Harus diisi apa: Nomor port (integer).
#      * Maksudnya: Port default untuk FastAPI/Uvicorn adalah 8000. Jika port 8000 sudah terpakai oleh aplikasi lain, Anda bisa menggantinya dengan port lain (misal: 8080, 5000, 3000, dll).
