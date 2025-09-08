# Modul 2: Memberi Jiwa pada Neuron - Proses Belajar

Di modul ini, kita mengubah neuron kita dari kalkulator statis menjadi entitas yang bisa belajar dan beradaptasi. Kita telah mengimplementasikan **Training Loop**, jantung dari semua *machine learning*.

### 1. Konsep Inti: Siklus Belajar (Training Loop)

Proses belajar adalah siklus yang berulang-ulang, terdiri dari:

1. **Tebak (Feedforward):** Neuron menerima input dan memberikan prediksi.
2. **Ukur Kesalahan (Loss Function):** Kita mengukur seberapa jauh prediksi dari jawaban yang benar. Dalam kode kita, `error = correct_answer - predicted_output`.
3. **Cari Arah Perbaikan (Gradient Descent):** Menggunakan Kalkulus untuk menentukan bagaimana cara mengubah `weights` dan `bias` untuk mengurangi *error*.
4. **Perbaiki (Update):** Memperbarui `weights` dan `bias` berdasarkan hasil dari Gradient Descent.
5. **Ulangi (Epochs):** Mengulangi seluruh proses ini ribuan kali. Satu kali melewati seluruh dataset disebut satu **epoch**.

### 2. Parameter Penting dalam Training

* **Learning Rate (Tingkat Belajar):** Mengontrol seberapa besar "langkah" perbaikan yang kita ambil setiap kali.
    * **Terlalu besar:** Bisa "melompati" solusi terbaik, seperti mencari lembah tapi melangkah terlalu lebar.
    * **Terlalu kecil:** Proses belajar akan sangat lambat.
    * Dalam kode kita: `learning_rate = 0.1`.

* **Epochs (Epoch):** Berapa kali kita mengulang siklus pelatihan pada keseluruhan dataset. Semakin banyak epoch, semakin banyak kesempatan neuron untuk belajar, tapi bisa menyebabkan *overfitting* (terlalu hafal dengan data training).
    * Dalam kode kita: `epochs = 10000`.

### 3. Anatomi Kode Baru: `neuron_belajar.py`

* **`_sigmoid_derivative(self, x)`:** Fungsi ini menghitung turunan (gradien) dari fungsi sigmoid. Ini adalah komponen Kalkulus yang krusial untuk *backpropagation*. Ia memberitahu kita seberapa "curam" kurva sigmoid di titik tertentu, yang menandakan seberapa besar pengaruh perubahan input terhadap output.

* **`train(...)`:** Metode ini adalah implementasi dari Training Loop. Perhatikan baris ini:
    `adjustment = learning_rate * error * self._sigmoid_derivative(predicted_output)`
    Inilah inti dari Gradient Descent dan Backpropagation dalam bentuk paling sederhana. Ia menghitung "penyesuaian" yang tepat berdasarkan tiga hal: seberapa besar langkahnya (`learning_rate`), seberapa salah tebakannya (`error`), dan seberapa yakin neuron dengan tebakannya (`_sigmoid_derivative`).

* **Normalisasi Data:** Perhatikan bahwa input data (`[0.2, 0.2]`, dll.) bukanlah tinggi dan berat asli. Data ini telah **dinormalisasi**. Ini adalah praktik standar dalam *machine learning* untuk membuat proses training lebih stabil dan cepat. Kita mengubah semua nilai ke dalam rentang yang serupa (misalnya -1 sampai 1).

### 4. Eksperimen Mandiri (Tugas Anda!)

Coba jalankan `neuron_belajar.py` dan perhatikan bagaimana *Total Error* menurun setiap 1000 epoch. Lalu, coba "rusak" lagi:

1. **Ubah Learning Rate:** Apa yang terjadi jika `learning_rate` diubah menjadi `1.0`? Atau menjadi `0.001`? Apakah trainingnya lebih cepat atau lebih lambat? Apakah hasilnya lebih akurat?
2. **Ubah Jumlah Epoch:** Coba latih hanya dengan `100` epoch. Apakah prediksinya masih akurat? Bagaimana jika `100000`?
3. **Beri Data Sulit:** Coba buat data tes baru yang ambigu, misalnya `new_data_3 = np.array([0.0, 0.0])` (Tinggi 150, Berat 50). Apa output yang diberikan neuron? Mengapa menurutmu begitu?

### 5. Apa Selanjutnya?

Selamat! Kamu telah berhasil membangun dan melatih unit kecerdasan buatan pertamamu dari NOL. Kamu sudah memahami konsep paling fundamental: **feedforward** dan **backpropagation**.

Langkah selanjutnya adalah beralih dari satu neuron ke **jaringan neuron (Neural Network)**. Kita akan menyusun banyak neuron menjadi beberapa lapisan (*layers*) untuk menyelesaikan masalah yang lebih kompleks, yang tidak mungkin dipecahkan oleh satu neuron saja.