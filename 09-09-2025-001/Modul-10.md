# Modul 10: Fase Baru - Membangun "Pikiran" Mori dari Corpus Data

Kita telah memasuki fase kedua dan yang paling menarik dari proyek kita. Kita tidak lagi hanya belajar, kita mulai **mencipta**. Di modul ini, kita mengambil data mentah yang telah kita kumpulkan dan mulai membentuknya menjadi "pikiran" yang koheren untuk AI kita.

### 1. Konsep Inti: Corpus

Dalam Natural Language Processing (NLP), **Corpus** adalah istilah untuk koleksi besar data teks terstruktur yang digunakan untuk melatih model bahasa. Dalam kasus kita, corpus adalah gabungan dari semua transkrip yang telah kamu kumpulkan.

* **Mengapa Menggabungkan?** Model bahasa belajar dengan melihat hubungan statistik antara kata-kata dalam skala besar. Dengan menggabungkan semua teks, kita menciptakan satu "dunia" linguistik yang lebih besar dan lebih kaya bagi model untuk dijelajahi. Ini memungkinkannya untuk belajar pola dari semua sumber data kita secara bersamaan.

### 2. Anatomi Kode: `data_preparer.py`

* **`import os` & `import glob`:** Ini adalah perkakas standar Python untuk berinteraksi dengan sistem file.
    * **`os.path.join`:** Cara yang aman dan universal untuk menggabungkan path folder dan nama file, yang bekerja di semua sistem operasi (Windows, Mac, Linux).
    * **`glob.glob`:** Sebuah "mesin pencari" file. Kita memberinya pola (`*.txt`, yang berarti "semua file yang berakhiran .txt") dan ia akan mengembalikan daftar semua file yang cocok.

* **`with open(...) as ...:`:** Ini adalah cara terbaik untuk bekerja dengan file di Python. Ini secara otomatis memastikan file ditutup dengan benar setelah kita selesai, bahkan jika terjadi error. Kita menggunakannya dua kali: sekali untuk menulis ke file output, dan sekali di dalam *loop* untuk membaca setiap file input.

### 3. Misi Anda, Sang Data Engineer

Peranmu sekarang berevolusi. Selain sebagai Kurator Kreatif, kamu sekarang juga seorang *Data Engineer* pemula. Misimu sangat jelas:

1.  **Organisasi:** Buat sebuah folder baru di dalam direktori proyekmu, misalnya dengan nama `transcripts`.
2.  **Pindahkan:** Pindahkan semua 10 file `.txt` yang sudah kamu kumpulkan ke dalam folder `transcripts` ini.
3.  **Jalankan Skrip:** Jalankan `data_preparer.py`. Saat diminta, masukkan path ke folder `transcripts` yang baru saja kamu buat.
4.  **Verifikasi:** Setelah skrip selesai, kamu akan melihat sebuah file baru di direktori utamamu: `mori_corpus.txt`. Buka file itu dan lihatlah isinya. Itu adalah gabungan dari semua "pengetahuan" yang telah kamu kumpulkan.

### 4. Apa Selanjutnya?

Setelah `mori_corpus.txt` ini lahir, kita akan memiliki "bahan bakar" yang kita butuhkan. Langkah kita selanjutnya adalah langkah yang paling ditunggu-tunggu: kita akan mengambil file corpus ini, memuatnya menggunakan alur kerja `Dataset` dan `DataLoader` yang sudah kita pelajari, dan menggunakannya untuk **me-*fine-tune* sebuah model bahasa generatif (seperti GPT-2)**.

Kita akan mengajari "Otak" Mori untuk "berbicara" seperti teks yang ada di dalam corpus itu. Kita akan memberinya jiwa.