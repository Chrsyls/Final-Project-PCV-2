# Kamen Rider Shape Body Tracking

**Kamen Rider Shape Body Tracking** adalah aplikasi Augmented Reality (AR) sederhana berbasis Python yang mengubah pengguna menjadi karakter bergaya Kamen Rider secara real-time menggunakan kamera web.

Proyek ini memanfaatkan **Google MediaPipe** untuk pelacakan tubuh (holistic tracking) dan **OpenCV** untuk menggambar overlay armor digital. Dilengkapi dengan algoritma smoothing (One Euro Filter) untuk mengurangi jitter (getaran) pada gerakan, serta fitur interaktif wajah.

## Fitur Utama

* **Full Body Armor Overlay:** Menggambar kerangka tubuh dengan warna khas (Hijau/Perak) secara dinamis mengikuti gerakan pengguna.
* **Interaksi Wajah:**
* **Mata:** Berubah warna menjadi hitam saat pengguna berkedip.
* **Mulut:** Pelat mulut (mouthplate) bergerak naik-turun saat pengguna membuka mulut.


* **Henshin Belt Animation:** Efek lampu berdenyut (pulsating) pada bagian pinggang yang mensimulasikan sabuk Henshin.
* **Smooth Movement:** Menggunakan implementasi **One Euro Filter** untuk memuluskan data koordinat landmark, sehingga gerakan terlihat lebih luwes dan tidak patah-patah.
* **Hand Tracking Persistence:** Fitur anti-disappear (caching) untuk mencegah jari-jari menghilang tiba-tiba saat pelacakan tangan tidak stabil atau keluar frame sejenak.

## Prasyarat Teknis

Pastikan kamu telah menginstal **Python 3.7** atau versi yang lebih baru. Proyek ini membutuhkan library berikut:

* `opencv-python` (Pengolahan citra)
* `mediapipe` (Machine Learning tracking)
* `numpy` (Operasi matriks matematika)

## Instalasi

1. **Clone atau Download** repositori/file ini ke komputer kamu.
2. **Siapkan Virtual Environment (Opsional tapi disarankan):**
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate

```


3. **Install Dependencies:**
Jalankan perintah berikut di terminal:
```bash
pip install opencv-python mediapipe numpy

```



## Cara Penggunaan

1. Pastikan webcam kamu terhubung.
2. Jalankan script utama:
```bash
python main.py

```


3. Sebuah jendela bernama **"Kamen Rider Shape Body Tracking"** akan muncul.
4. Berdirilah di depan kamera agar seluruh tubuh terlihat untuk hasil terbaik.
5. **Kontrol:**
* Gerakkan tubuh untuk menggerakkan karakter.
* Berkedip atau buka mulut untuk melihat efek wajah.
* Tekan tombol **`ESC`** pada keyboard untuk menutup aplikasi.



## Konfigurasi (Customization)

Kamu dapat mengubah beberapa parameter di bagian atas file `main.py` untuk menyesuaikan tampilan:

```python
# ================= CONFIG =================
WINDOW_NAME = "Kamen Rider Shape Body Tracking"

# Sensitivitas deteksi wajah
BLINK_RATIO_THRESHOLD = 5.0  # Semakin kecil, semakin mudah terdeteksi kedipan
MOUTH_OPEN_THRESHOLD = 0.5   # Semakin kecil, semakin sensitif deteksi mulut buka

# Skema Warna (Format BGR: Blue, Green, Red)
COLOR_MAIN = (0, 170, 0)      # Warna utama armor (Hijau)
COLOR_DARK = (0, 90, 0)       # Warna aksen gelap
COLOR_SILVER = (200, 200, 200) # Warna persendian/perak
COLOR_EYE = (0, 0, 255)       # Warna mata (Merah)

```

## Struktur Kode

* **Inisialisasi MediaPipe:** Mengaktifkan modul `Pose`, `FaceMesh`, dan `Hands`.
* **Class `OneEuro`:** Algoritma filtering matematika untuk menyaring noise frekuensi tinggi dari data sensor/kamera.
* **Loop Utama:**
1. Membaca frame dari webcam.
2. Memproses deteksi AI (Pose, Face, Hand).
3. Menghitung logika interaksi (Kedip, Mulut, Sabuk).
4. Menggambar garis dan bentuk (Line & Circle) menggunakan `cv2`.
5. Menampilkan hasil.



## Troubleshooting

* **Error: `Module not found**`: Pastikan kamu sudah menjalankan `pip install ...` sesuai panduan instalasi.
* **Lag / Lambat**:
* Coba kurangi resolusi kamera di kode (`cap.set(3, 640)`, `cap.set(4, 480)`).
* Pastikan pencahayaan ruangan cukup terang agar MediaPipe dapat mendeteksi tubuh lebih cepat.


* **Tangan Berkedip-kedip**: Pastikan tangan tidak terlalu dekat dengan tepi layar atau saling menutupi.

## Lisensi

Proyek ini dibuat untuk tujuan edukasi dan hobi. Bebas untuk dimodifikasi dan dikembangkan lebih lanjut.

---

Created for Tokusatsu Fans.
