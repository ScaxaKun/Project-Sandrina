# Activity Monitoring System

## 1. Deskripsi Proyek

Activity Monitoring System merupakan prototipe sistem pemantauan aktivitas individu berbasis _Computer Vision_. Sistem dikembangkan menggunakan Python untuk mendeteksi manusia, melacak individu, mengklasifikasikan aktivitas, menghitung durasi aktivitas, dan memberikan peringatan melalui dashboard monitoring.

Sistem saat ini menggunakan webcam sebagai sumber video.

## 2. Fitur Sistem

- Deteksi manusia menggunakan YOLO.
- Pelacakan individu menggunakan ByteTrack.
- Klasifikasi aktivitas: sitting, standing, dan walking.
- Penghitungan durasi aktivitas berdasarkan ID tracking.
- Peringatan ketika durasi aktivitas mencapai batas yang ditentukan.
- Dashboard monitoring dan riwayat aktivitas.

## 3. Teknologi yang Digunakan

- Python
- OpenCV
- Ultralytics YOLO
- ByteTrack
- NumPy
- Pillow

## 4. Struktur Folder

```text
Activity_Monitoring_Project/
├── activity_monitor.py
├── requirements.txt
├── README.md
├── yolo26n.pt
└── runs/
    └── polar_3class/
        └── weights/
            └── best.pt
```

## 5. Persyaratan Sistem

Sebelum menjalankan program, pastikan:

- Python telah terinstal pada komputer.
- Webcam tersedia dan dapat digunakan.
- Seluruh file model tersedia pada folder proyek.
- Komputer terhubung ke internet untuk instalasi library.

## 6. Persiapan Environment

Buka terminal pada folder proyek.

Buat virtual environment menggunakan perintah:

```bash
python -m venv venv
```

Aktifkan virtual environment menggunakan perintah berikut pada Windows PowerShell:

```powershell
.\venv\Scripts\activate
```

Setelah virtual environment aktif, instal library yang diperlukan:

```bash
pip install -r requirements.txt
```

Tunggu hingga proses instalasi selesai.

## 7. Menjalankan Program

Pastikan virtual environment telah aktif dan webcam terhubung.

Jalankan program menggunakan perintah:

```bash
python activity_monitor.py
```

Program akan menampilkan dashboard monitoring dan mulai memproses video dari webcam.

## 8. Kontrol Program

- **H**: Menampilkan atau menyembunyikan riwayat aktivitas.
- **P**: Menampilkan atau menyembunyikan halaman peringatan.
- **Q**: Mengakhiri program.
