# Digital Image Watermarking berbasis DCT dan Redundansi Spasial

Repositori ini mengimplementasikan algoritma watermarking citra digital secara empiris. Sistem menyisipkan watermark citra biner ke dalam foto inang berwarna. Algoritma menggunakan teknik Discrete Cosine Transform (DCT) dan Quantization Index Modulation (QIM) pada pita frekuensi menengah. Kode program menerapkan metode redundansi spasial dan pemungutan suara terbanyak (*majority voting*). Mekanisme terpadu ini meningkatkan ketahanan matriks data secara sistematis terhadap serangan kompresi JPEG.

## Fitur Utama

* **Ekstraksi Buta (*Blind Extraction*):** Sistem mengekstrak informasi watermark tanpa memerlukan citra inang asli.
* **Isolasi Saluran Luminans:** Algoritma mengonversi ruang warna BGR menjadi YCrCb. Sistem membatasi modifikasi koefisien secara eksklusif pada saluran Y untuk menjaga integritas visual warna.
* **Redundansi Spasial:** Kode menduplikasi bit watermark ke dalam berbagai blok spasial 8x8 secara matematis.
* **Ketahanan Mayoritas:** Fungsi ekstraksi mengakumulasi probabilitas bit menggunakan metode *majority voting*. Metode ini menekan tingkat *Bit Error Rate* (BER) pada skenario kompresi tingkat tinggi.
* **Evaluasi Otomatis:** Program menyimulasikan kompresi JPEG dari tingkat Quality Factor (QF) 10 hingga 100. Skrip menghitung nilai *Peak Signal-to-Noise Ratio* (PSNR), BER, dan *Normalized Correlation* (NC) secara otomatis.

## Struktur Repositori

* `watermarkDCT_Biner.py`: Skrip utama untuk proses penyisipan, ekstraksi, dan pengujian kualitas.
* `requirements.txt`: Daftar pustaka dependensi komputasi standar.
* `Asset/`: Direktori penyimpanan sampel foto inang dan citra watermark biner.
* `Result/`: Direktori penyimpanan hasil grafik evaluasi dan komparasi visual.
* `.gitignore`: Konfigurasi untuk memblokir pelacakan file residu sementara.

## Kebutuhan Sistem

Anda wajib menginstal pustaka komputasi Python berikut sebelum menjalankan program:

```bash
pip install -r requirements.txt

Panduan Penggunaan
Anda mengeksekusi skrip ini melalui antarmuka baris perintah (command line). Anda memiliki dua opsi eksekusi operasional.

1. Eksekusi Simulasi Demo (Default):
Sistem menghasilkan matriks gambar inang acak dan pola watermark bawaan jika Anda tidak memberikan argumen rute file.

Bash
python watermarkDCT_Biner.py
2. Eksekusi Data Kustom:
Anda memanggil gambar wajah asli dan citra biner kustom dari dalam direktori Asset.

Bash
python watermarkDCT_Biner.py Asset/foto_wajah.jpg Asset/watermark_biner.png
Hasil Keluaran Empiris
Sistem mengekspor dua file representasi visual ke dalam folder Result/ secara otomatis:

perbandingan_foto.png: Visualisasi komparasi kualitas struktural gambar inang sebelum dan sesudah penyisipan matriks watermark.

evaluasi_watermark.png: Grafik linier yang menampilkan korelasi empiris antara skalar Quality Factor (QF) JPEG dengan nilai BER, NC, dan PSNR.

Informasi Penulis
Nama: Muhamad Radhitya Alamsyah

NIM: 18224126
