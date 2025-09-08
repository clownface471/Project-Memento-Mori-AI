# Modul 9: Babak Baru - Membangun Perpustakaan Memori Memento Mori

Selamat datang di babak kedua dari perjalanan kita! Fondasi telah selesai. Mulai sekarang, setiap baris kode yang kita tulis akan secara langsung berkontribusi pada penciptaan VTuber AI "Memento Mori".

### 1. Konsep Inti: Data Adalah Darah Kehidupan AI

Model *pre-trained* memberi kita "otak" yang kosong tapi cerdas. **Data yang kita berikan padanya saat *fine-tuning* akan menjadi "jiwa", "kenangan", dan "kepribadian"-nya.** Kualitas dari AI kita akan 100% ditentukan oleh kualitas data yang kita kumpulkan. Inilah sebabnya mengapa langkah pertama kita adalah membangun alat untuk mengumpulkan data, bukan langsung melatih model.

### 2. Anatomi Kode: `youtube_scraper.py`

* **`youtube-transcript-api`:** *Library* ini adalah jembatan kita ke arsip data YouTube. Ia menangani semua detail teknis yang rumit untuk menemukan dan mengunduh data transkrip/subtitle.
* **`try...except` Block:** Ini adalah praktik *error handling* yang sangat penting. Kita tidak bisa berasumsi semua video punya transkrip. Blok ini memastikan program kita tidak *crash* jika video target tidak valid atau transkripnya tidak tersedia, melainkan memberikan pesan yang jelas kepada pengguna.
* **`" ".join([segment['text'] for segment in transcript_list])`:** Baris ini mungkin terlihat rumit, tapi ini adalah cara Python yang sangat efisien untuk melakukan tiga hal:
    1.  `for segment in transcript_list`: Ulangi setiap bagian dari transkrip.
    2.  `segment['text']`: Dari setiap bagian, ambil hanya teksnya (abaikan timestamp, dll.).
    3.  `" ".join(...)`: Gabungkan semua potongan teks itu menjadi satu paragraf panjang, dengan spasi sebagai pemisah.
* **`if __name__ == "__main__":`:** Ini adalah konvensi standar dalam Python. Kode di dalam blok ini hanya akan berjalan jika kita menjalankan file ini secara langsung (bukan saat diimpor oleh file lain). Ini membuat kode kita bisa digunakan kembali (reusable) di masa depan.

### 3. Misi Anda, Sang Kurator Kreatif

Sekarang, peranmu sebagai *Creative Director* dimulai. *Tool* ini adalah "jaring"-mu. Misimu adalah:

1.  **Cari "Bahan Bacaan" untuk Mori:** Jelajahi YouTube dan temukan **3-5 video** yang gaya bicaranya sesuai dengan kepribadian Memento Mori. Carilah video yang:
    * **Filosofis atau Puitis:** Sesuai dengan sisi dalamnya yang merenung.
    * **Gameplay Cerdas tapi Kacau:** Menunjukkan sisi "topeng"-nya yang ceria tapi juga perjuangannya.
    * **Storytelling atau Esai Video:** Mencerminkan keahliannya sebagai penulis.
2.  **Jalankan Skrip:** Gunakan skrip `youtube_scraper.py` ini untuk mengunduh transkrip dari video-video tersebut.
3.  **Kumpulkan Hasilnya:** Di akhir misi ini, kamu seharusnya memiliki 3-5 file `.txt` yang akan menjadi "buku-buku" pertama di perpustakaan Mori.

Kumpulan file teks inilah yang akan kita gunakan di langkah selanjutnya untuk membuat `Dataset` kustom pertama kita dan mulai proses *fine-tuning* untuk **generasi teks**.

Perburuan dimulai!