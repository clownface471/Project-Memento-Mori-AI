# Modul 1: Membangun "Atom" Kecerdasan Buatan - Neuron Pertama Anda

Selamat datang di modul pertama! Tujuan kita di sini adalah memahami blok bangunan paling fundamental dari semua AI modern: **Neuron**. Anggap ini sebagai satu buah "Lego" yang nantinya bisa kita susun untuk membangun struktur yang luar biasa kompleks seperti LLM.

### 1. Konsep Inti: Si Pembuat Keputusan Sederhana

Sebuah neuron buatan meniru cara kerja neuron biologis secara sederhana. Ia adalah unit pembuat keputusan kecil. Ingat analogi "pergi ngopi"?

* **Inputs:** Data mentah yang diterima (misal: cuaca, punya payung, teman ikut). Dalam kode, ini adalah array NumPy `[0, 1, 1]`.
* **Weights (Bobot):** Tingkat kepentingan dari setiap input. Input yang lebih penting punya bobot lebih tinggi. Dalam kode, ini adalah array `[0.5, 0.8, 1.5]`.
* **Bias:** Kecenderungan bawaan dari neuron untuk aktif atau tidak aktif, bahkan sebelum melihat input. Dalam kode, ini adalah angka tunggal `-1.0`.
* **Activation Function (Fungsi Aktivasi):** "Tombol" terakhir yang menentukan output akhir, biasanya dalam rentang nilai tertentu (misal, 0 sampai 1) untuk memudahkan interpretasi.

### 2. Anatomi Kode: Membedah `neuron.py`

Mari kita bedah kode yang sudah kita buat.

* **`class Neuron:`**
    Ini adalah 'cetak biru' kita. Semua neuron yang kita buat akan mengikuti struktur ini.
* **`__init__(self, jumlah_input):`**
    Ini adalah "konstruktor". Fungsi ini berjalan secara otomatis setiap kali kita membuat neuron baru (`neuron_kopi = Neuron(3)`). Tugasnya adalah menyiapkan `weights` dan `bias` awal.
* **`_sigmoid(self, x):`**
    Ini adalah fungsi aktivasi yang kita pilih. Namanya Sigmoid karena grafiknya berbentuk seperti huruf 'S'. Fungsi ini mengambil angka apa pun (negatif atau positif) dan mengubahnya menjadi angka antara 0 dan 1.
    Rumus: $ \sigma(x) = \frac{1}{1 + e^{-x}} $
* **`feedforward(self, inputs):`**
    Ini adalah proses "berpikir" satu arah (dari input ke output).
    1.  `total = np.dot(inputs, self.weights) + self.bias`: Ini adalah langkah kalkulasi inti. Ia menggabungkan semua input dan bobotnya, lalu menambahkan bias.
    2.  `return self._sigmoid(total)`: Hasil kalkulasi tadi kemudian "ditekan" oleh fungsi sigmoid untuk menghasilkan output akhir.

### 3. Matematika di Balik Sihir

* **Algebra Linear:** Perintah `np.dot(inputs, self.weights)` adalah implementasi langsung dari operasi **Dot Product**. Ini adalah jantung dari komputasi *deep learning* karena sangat efisien.
* **Kalkulus:** (Akan kita bahas di modul selanjutnya) Kalkulus akan menjadi kunci untuk proses "belajar", di mana neuron secara otomatis menyesuaikan `weights` dan `bias`-nya untuk mengurangi kesalahan.

### 4. Eksperimen Mandiri (Tugas Anda!)

Cara terbaik untuk belajar adalah dengan mencoba. Buka file `neuron.py` dan coba ubah beberapa hal:

1.  **Ubah Skenario:** Apa yang terjadi jika inputnya `[1, 0, 0]` (Hujan, tidak ada payung, tidak ada teman)? Prediksi dulu hasilnya, lalu jalankan kodenya. Apakah sesuai dengan intuisimu?
2.  **Ubah Kepribadian Neuron:** Apa yang terjadi jika kamu mengubah `bias` menjadi `1.0` (menjadi orang yang sangat rajin)? Bagaimana outputnya berubah untuk skenario yang sama?
3.  **Ubah Prioritas:** Bagaimana jika bobot untuk "teman ikut" kamu turunkan menjadi `0.1` dan bobot "hujan" kamu naikkan menjadi `2.0`?

### 5. Apa Selanjutnya?

Saat ini, neuron kita masih "bodoh". Kita yang menentukan bobot dan biasnya. Di modul selanjutnya, kita akan masuk ke bagian paling ajaib: **Training (Pelatihan)**. Kita akan mengajarkan neuron cara untuk menemukan bobot dan bias terbaiknya sendiri dengan melihat contoh. Kita akan membahas konsep **Loss Function**, **Gradient Descent**, dan **Backpropagation**.