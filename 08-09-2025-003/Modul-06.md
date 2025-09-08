# Modul 6: Jembatan ke Bahasa - Tokenisasi & Klasifikasi Teks

Kita berada di langkah terakhir sebelum mulai mengerjakan "Otak" VTuber kita. Di modul ini, kita mengadaptasi alur kerja PyTorch yang sudah kita kuasai dari dunia gambar ke dunia bahasa.

### 1. Konsep Inti: Tokenisasi

Komputer tidak memahami teks, ia memahami angka. **Tokenisasi** adalah proses mengubah teks (string) menjadi urutan angka (token ID).

* **Tokenizer:** Alat yang berisi "kamus" (vocabulary) dan aturan untuk melakukan konversi ini.
* **Hugging Face `transformers`:** *Library* standar industri yang menyediakan akses ke *tokenizer* dan model yang sudah di-*pre-train* pada data berskala masif. Kita menggunakan `BertTokenizer` sebagai contoh.
* **Proses `tokenizer.encode_plus`:** Fungsi serbaguna ini melakukan banyak hal:
    * Mengubah kata menjadi token ID.
    * Menambahkan token spesial seperti `[CLS]` (Classify) di awal dan `[SEP]` (Separator) di akhir.
    * Membuat semua urutan token memiliki panjang yang sama dengan **padding** (menambah token `[PAD]`) atau **truncation** (memotong). Ini krusial agar bisa diproses dalam *batch*.
    * Membuat **attention mask**, yang memberitahu model token mana yang asli dan mana yang hanya *padding*.

### 2. Anatomi Kode: `pytorch_text.py`

* **Custom `Dataset`:** Berbeda dengan MNIST di mana `Dataset` sudah disediakan, untuk data teks kita seringkali perlu membuatnya sendiri. `SentimentDataset` adalah contoh sempurna. Ia membungkus data mentah kita (kalimat & label) dan melakukan tokenisasi "on-the-fly" setiap kali `DataLoader` meminta sebuah item.
* **`nn.Embedding` Layer:** Ini adalah lapisan ajaib pertama dalam model NLP. Tugasnya adalah mengambil ID token (sebuah angka sederhana, misal: 582) dan mengubahnya menjadi sebuah **vektor padat (dense vector)** yang kaya akan makna (misal, sebuah array dengan 128 angka). Vektor ini disebut *embedding*. Kata-kata dengan makna serupa akan memiliki vektor yang mirip. Lapisan ini **dipelajari** selama training.
* **Model Sederhana:** Model kita mengambil *embedding* dari semua token dalam satu kalimat, merata-ratakannya untuk mendapatkan satu representasi tunggal untuk seluruh kalimat, lalu memasukkannya ke lapisan `Linear` biasa untuk klasifikasi.
* **Training Loop yang Sama:** Perhatikan! Setelah data berhasil dimuat oleh `DataLoader`, *training loop*-nya hampir identik dengan yang kita gunakan untuk MNIST. Ini menunjukkan betapa kuat dan fleksibelnya alur kerja ini.

### 3. Apa Selanjutnya?

Kamu baru saja berhasil membangun dan melatih sebuah **pengklasifikasi teks** dari nol. Kamu sudah menguasai alur kerja data untuk gambar dan teks. Fondasi kita sudah lengkap.

Langkah berikutnya adalah langkah yang paling kita tunggu-tunggu. Kita akan berhenti membangun model kecil dari nol. Kita akan belajar cara **mengambil model bahasa raksasa yang sudah di-*pre-train*** (seperti versi kecil dari GPT atau BERT), memuatnya menggunakan Hugging Face `transformers`, dan melakukan **fine-tuning**: melatihnya lagi sedikit pada dataset spesifik kita (misalnya, data sentimen yang lebih besar, atau nanti, transkrip *stream*) untuk mengajarinya tugas baru atau memberinya "kepribadian" baru.

Ini adalah gerbang menuju "Otak" VTuber kita.