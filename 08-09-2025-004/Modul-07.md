# Modul 7: Kekuatan Pre-training & Hugging Face Pipeline

Di modul ini, kita mengalami lompatan kuantum. Kita berhenti sejenak membangun model dari nol dan mulai menggunakan hasil kerja dari seluruh industri AI: **model yang sudah di-*pre-train***.

### 1. Konsep Inti: Pre-training vs. Fine-tuning

Ini adalah paradigma fundamental dalam NLP modern.

* **Pre-training:** Proses melatih model bahasa raksasa pada data teks dalam jumlah masif (misal: seluruh Wikipedia, semua buku digital, triliunan halaman web). Tujuannya adalah agar model "memahami" bahasa secara umum: tata bahasa, fakta, nuansa, dan kemampuan menalar sederhana. Ini seperti seorang manusia yang menempuh pendidikan S1, S2, hingga S3 dalam studi umum. Proses ini sangat mahal dan hanya bisa dilakukan oleh perusahaan besar.
* **Fine-tuning:** Proses mengambil model yang sudah di-*pre-train* ("lulusan S3") dan melatihnya lagi sedikit pada dataset yang lebih kecil dan spesifik untuk tugas tertentu. Ini seperti memberi seorang sarjana pelatihan kerja untuk peran spesifik di perusahaanmu. Ini jauh lebih murah, lebih cepat, dan merupakan cara standar untuk membangun aplikasi AI modern.

### 2. Anatomi Kode: `hf_pipeline.py`

* **`pipeline("sentiment-analysis")`:** Ini adalah "jalan pintas ajaib" dari Hugging Face. Perintah ini melakukan banyak hal di belakang layar:
    1.  Menemukan model terbaik yang tersedia di Hugging Face Hub untuk tugas "sentiment-analysis".
    2.  Mengunduh arsitektur model tersebut.
    3.  Mengunduh bobot (weights) yang sudah di-*pre-train* dan di-*fine-tune*.
    4.  Mengunduh *tokenizer* yang sesuai.
    5.  Membungkus semuanya dalam satu objek yang sangat mudah digunakan.
* **Tidak Ada Training Loop:** Perhatikan, tidak ada `DataLoader`, tidak ada `optimizer`, tidak ada `loss.backward()`. Kita tidak melatih apa-apa. Kita hanya menggunakan model yang **sudah jadi** untuk melakukan inferensi (prediksi).

### 3. Apa Selanjutnya?

Kamu baru saja menyaksikan kekuatan sebenarnya dari model bahasa modern. Model yang kita panggil melalui `pipeline` telah dilatih pada data jutaan kali lebih banyak daripada yang bisa kita kumpulkan sendiri.

Tentu saja, model generalis ini mungkin tidak akan mengerti slang VTuber atau konteks spesifik dari game yang sedang kita mainkan. Di sinilah langkah kita selanjutnya masuk.

Setelah ini, kita akan belajar **cara melakukan *fine-tuning* secara manual**. Kita akan mengambil model *pre-trained* (seperti `bert-base-uncased`), menggabungkannya dengan alur kerja `Dataset` dan `DataLoader` yang sudah kita pelajari, dan melatihnya pada dataset sentimen yang lebih besar. Ini akan memberi kita kontrol penuh untuk "membentuk" kepribadian model sesuai keinginan kita, langkah krusial untuk membangun "Otak" VTuber.