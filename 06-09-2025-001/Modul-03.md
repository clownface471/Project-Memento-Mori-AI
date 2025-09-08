# Modul 3: Membangun "Pasukan" - Jaringan Saraf Tiruan Pertama Anda

Selamat! Kita telah naik level dari membangun satu "prajurit" (Neuron) menjadi membangun sebuah "pasukan" (Neural Network). Ini adalah lompatan besar yang memungkinkan kita menyelesaikan masalah yang jauh lebih kompleks.

### 1. Konsep Inti: Kekuatan Lapisan Tersembunyi (Hidden Layer)

* **Batasan Neuron Tunggal:** Satu neuron hanya bisa belajar pemisah linear (seperti garis lurus). Ia tidak bisa menyelesaikan masalah yang "tidak lurus" (non-linear) seperti XOR.
* **Arsitektur Jaringan:** Dengan menambahkan setidaknya satu *Hidden Layer* di antara *Input* dan *Output*, kita memberikan kemampuan pada jaringan untuk belajar hubungan yang non-linear.
* **Cara Kerja:** Setiap neuron di *hidden layer* belajar fitur sederhana. Lapisan berikutnya (output layer) kemudian belajar dari *kombinasi* fitur-fitur tersebut untuk membuat keputusan yang lebih kompleks. Ini memungkinkan jaringan untuk "membengkokkan" garis keputusan dan menyesuaikan diri dengan pola data yang rumit.

### 2. Anatomi Kode: `neural_network.py`

Perbedaan utama dari kode sebelumnya terletak pada skala dan proses *backpropagation*.

* **`__init__(self)`:**
  Perhatikan bagaimana kita sekarang menginisialisasi **dua set bobot** (`weights_h` dan `weights_o`) dan **dua set bias** (`bias_h` dan `bias_o`). Bentuk (shape) dari matriks-matriks ini harus ditentukan dengan sangat hati-hati agar aljabar linearnya berjalan mulus.

* **`feedforward(self, x)`:**
  Prosesnya sekarang dua langkah:
  1. Dari input, melewati `weights_h` dan `bias_h`, diaktivasi oleh sigmoid, menghasilkan `hidden_output`.
  2. `hidden_output` tersebut menjadi input untuk langkah berikutnya, melewati `weights_o` dan `bias_o`, diaktivasi lagi, menghasilkan `final_output`.

* **`train(...)` dan Backpropagation:**
  Ini adalah bagian yang paling rumit. Prosesnya sekarang juga dua langkah, tapi berjalan **mundur**:
  1. Pertama, kita hitung *error* di lapisan output, persis seperti sebelumnya.
  2. Kemudian, kita **propagasi-mundurkan** *error* tersebut ke *hidden layer*. Intinya, kita bertanya: "Seberapa besar kontribusi setiap neuron di *hidden layer* terhadap *error* di output?" Ini dihitung menggunakan `np.dot(output_delta, self.weights_o.T)`.
  3. Setelah kita tahu "kesalahan" dari setiap lapisan, barulah kita bisa memperbarui bobot dan bias untuk kedua lapisan tersebut.

### 3. Studi Kasus: Mengapa XOR?

XOR adalah "Hello, World!" untuk *neural network* karena ia adalah contoh paling sederhana dari masalah yang **tidak dapat dipisahkan secara linear**. Jika modelmu bisa menyelesaikan XOR, itu membuktikan bahwa model tersebut mampu belajar pola non-linear.

### 4. Eksperimen Mandiri (Tugas Anda!)

1. **Ubah Arsitektur:** Di dalam `__init__`, coba ubah jumlah neuron di *hidden layer*. Misalnya, dari 2 menjadi 4. Apakah trainingnya menjadi lebih cepat atau lebih akurat?
2. **Ubah Hiperparameter:** Mainkan `learning_rate` dan `epochs`. Apa yang terjadi jika `learning_rate` terlalu tinggi untuk masalah ini?
3. **Hancurkan Jaringan:** Coba hapus *hidden layer*. Ubah kodenya agar input langsung terhubung ke output (seperti neuron tunggal kita sebelumnya). Apakah ia berhasil menyelesaikan XOR? (Petunjuk: Seharusnya tidak.)

### 5. Apa Selanjutnya?

Kamu telah membangun sebuah *Neural Network* fungsional dari nol hanya dengan NumPy. Ini adalah pencapaian yang sangat besar dan memberikanmu pemahaman fundamental yang tidak dimiliki banyak orang.

Langkah selanjutnya adalah menyadari mengapa kita butuh *framework* seperti **PyTorch** atau **TensorFlow**. Kamu lihat betapa rumitnya menulis *backpropagation* secara manual? *Framework* ini akan melakukan semua itu secara otomatis untuk kita, sehingga kita bisa fokus pada merancang arsitektur jaringan yang lebih besar dan lebih menarik, seperti yang digunakan dalam LLM.