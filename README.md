Panduan Penggunaan
Anda mengeksekusi skrip ini melalui antarmuka baris perintah (command line). Anda memiliki dua opsi eksekusi operasional.

1. Eksekusi Simulasi Demo (Default):
Sistem menghasilkan matriks gambar inang acak dan pola watermark bawaan jika Anda tidak memberikan argumen rute file.

Bash
python watermarkDCT_Biner.py
2. Eksekusi Data Kustom:
Anda memanggil gambar wajah asli dan citra biner kustom dari dalam direktori assets.

Bash
python watermarkDCT_Biner.py assets/foto_wajah.jpg assets/watermark_biner.png
Hasil Keluaran Empiris
Sistem mengekspor dua file representasi visual ke dalam folder results/ secara otomatis:

perbandingan_foto.png: Visualisasi komparasi kualitas struktural gambar inang sebelum dan sesudah penyisipan matriks watermark.

evaluasi_watermark.png: Grafik linier yang menampilkan korelasi empiris antara skalar Quality Factor (QF) JPEG dengan nilai BER, NC, dan PSNR.

Informasi Penulis
Nama: Muhamad Radhitya Alamsyah

NIM: 18224126
