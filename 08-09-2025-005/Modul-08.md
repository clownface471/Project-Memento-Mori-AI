# Modul 8: Puncak Fondasi - Fine-Tuning Model Pre-trained

Selamat! Kamu telah tiba di puncak dari perjalanan fondasi kita. Di modul ini, kita menggabungkan semua yang telah kita pelajari untuk melakukan **fine-tuning**, teknik paling kuat dan paling umum digunakan untuk membangun aplikasi AI modern.

### 1. Konsep Inti: Transfer Learning

*Fine-tuning* adalah sebuah bentuk dari **Transfer Learning**. Idenya adalah pengetahuan yang dipelajari oleh sebuah model untuk satu tugas (misalnya, memahami bahasa Inggris secara umum) bisa ditransfer dan digunakan untuk mempercepat pembelajaran pada tugas kedua yang berhubungan (misalnya, klasifikasi sentimen). Ini jauh lebih efisien daripada belajar dari nol.

### 2. Anatomi Kode: `pytorch_finetune.py`

* **`AutoTokenizer` & `AutoModelForSequenceClassification`:** Ini adalah "pintu gerbang" kita ke dunia Hugging Face. `Auto...` secara cerdas akan memuat arsitektur dan bobot yang tepat berdasarkan `MODEL_NAME` yang kita berikan. `...ForSequenceClassification` secara otomatis menambahkan "kepala" klasifikasi di atas model dasar, siap untuk kita latih.

* **`datasets.load_dataset`:** *Library* `datasets` adalah cara standar untuk mengakses ribuan dataset yang sudah siap pakai. Ia menangani pengunduhan, *caching*, dan menyediakan API yang sangat mudah untuk memproses data.

* **`.map(tokenize_function, batched=True)`:** Ini adalah cara yang sangat efisien untuk melakukan pra-pemrosesan. Ia akan mengambil fungsi `tokenize_function` kita dan menerapkannya pada seluruh dataset dalam *batch*, bahkan bisa menggunakan beberapa inti CPU untuk mempercepatnya.

* **Training Loop yang Lebih Sederhana:** Perhatikan betapa elegannya *training loop* kita sekarang.
    * `outputs = model(**batch)`: Kita bisa langsung memasukkan seluruh *batch* (yang merupakan *dictionary*) ke dalam model. Hugging Face menangani pemetaan `input_ids`, `attention_mask`, dll., secara otomatis.
    * `loss = outputs.loss`: Model Hugging Face bahkan secara otomatis menghitung *loss* untuk kita jika kita memberikan `labels`. Kita tidak perlu lagi memanggil `loss_function` secara terpisah.

* **`AdamW` Optimizer:** Ini adalah varian dari `Adam` yang sedikit dimodifikasi dan menjadi standar untuk me-*fine-tune* model Transformer.

### 3. Apa Selanjutnya? Inilah Awal yang Sebenarnya.

Kamu telah menyelesaikan seluruh fondasi. Kamu sudah bisa:
1.  Memahami cara kerja *neural network* dari nol (NumPy).
2.  Membangun arsitektur kompleks dengan *framework* modern (PyTorch).
3.  Mengelola alur kerja data untuk gambar dan teks (`DataLoader`).
4.  Menggunakan dan me-*fine-tune* model *pre-trained* berskala besar (Hugging Face).

Kamu **SIAP** untuk mulai membangun "Otak" VTuber kita.

Langkah selanjutnya dalam proyek kita adalah:
1.  **Mengumpulkan Data:** Mengumpulkan transkrip *stream*, chat, dan skrip yang sesuai dengan kepribadian VTuber yang kamu inginkan.
2.  **Menyiapkan Dataset:** Membuat *custom dataset* dari data tersebut, persis seperti yang kita lakukan hari ini.
3.  **Fine-tuning untuk Kepribadian:** Me-*fine-tune* sebuah model bahasa dasar (seperti `distilbert` atau `gpt2`) pada data kepribadianmu, bukan untuk klasifikasi, tapi untuk **generasi teks (text generation)**.

Jalankan kode `pytorch_finetune.py`. Prosesnya akan memakan waktu karena ia mengunduh dataset dan model yang cukup besar. Perhatikan bagaimana ia belajar, dan yang terpenting, lihat apakah prediksinya pada kalimat-kalimat sulit kita menjadi lebih baik.