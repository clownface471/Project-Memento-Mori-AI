# Modul 4: Naik Kelas - Membangun Jaringan dengan PyTorch

Selamat datang di bab baru! Kita meninggalkan implementasi manual dengan NumPy dan beralih ke *framework* industri: **PyTorch**. Tujuannya bukan untuk melupakan apa yang telah kita pelajari, tapi untuk menggunakan pemahaman fundamental itu saat bekerja dengan alat yang lebih kuat.

### 1. Mengapa Pindah dari NumPy?

Membangun dari nol dengan NumPy memberi kita pemahaman yang tak ternilai. Tapi itu **tidak efisien** dan **rentan error**. Bayangkan membangun LLM dengan jutaan neuron; menulis *backpropagation* manual akan menjadi mimpi buruk.

*Framework* seperti PyTorch menyediakan **abstraksi**. Ia menangani bagian-bagian yang rumit dan repetitif, sehingga kita bisa fokus pada bagian yang kreatif: **merancang arsitektur jaringan**.

### 2. Konsep Inti PyTorch

* **Tensors:** Anggap ini sebagai `ndarray` dari NumPy, tapi dengan kekuatan super. Tensor bisa melacak operasi yang terjadi padanya dan bisa dengan mudah dipindahkan ke GPU untuk komputasi super cepat.
* **`nn.Module`:** Semua model *neural network* di PyTorch adalah sebuah `class` yang mewarisi (`inherit`) dari `nn.Module`. Ini memberi kita struktur dan fungsionalitas dasar.
* **Loss Functions (e.g., `nn.MSELoss`):** PyTorch sudah menyediakan puluhan implementasi fungsi loss yang sudah teroptimasi. Kita tidak perlu lagi menghitung `error = y_true - y_pred` secara manual.
* **Optimizers (e.g., `torch.optim.SGD`):** Ini adalah implementasi dari algoritma *gradient descent*. Tugasnya adalah mengambil gradien yang dihitung dan memperbarui bobot model kita secara otomatis. Kita tidak perlu lagi menulis `self.weights += ...`.
* **Autograd Engine (`loss.backward()`):** Ini adalah **jantung dari PyTorch**. Saat kita memanggil `.backward()` pada sebuah `loss`, PyTorch akan secara otomatis menghitung turunan (*gradient*) dari *loss* tersebut terhadap setiap parameter di dalam model kita. Inilah yang menggantikan semua perhitungan `_sigmoid_derivative` dan *chain rule* manual kita.

### 3. Anatomi Kode: `pytorch_xor.py`

Lihat betapa berbedanya siklus pelatihan kita:

**Versi NumPy (Manual):**
1. Hitung `predicted_output`.
2. Hitung `error`.
3. Hitung `adjustment` menggunakan turunan manual.
4. Perbarui `weights` secara manual.
5. Perbarui `bias` secara manual.

**Versi PyTorch (Otomatis):**
1. `optimizer.zero_grad()` (Bersihkan sisa perhitungan lama).
2. `outputs = model(X)` (Dapatkan prediksi).
3. `loss = loss_function(outputs, y)` (Hitung loss).
4. `loss.backward()` (Hitung semua gradien secara ajaib).
5. `optimizer.step()` (Perbarui semua bobot secara ajaib).

Lima baris kode ini adalah inti dari hampir semua skrip training *deep learning* modern.

### 4. Apa Selanjutnya?

Sekarang kita memiliki perkakas yang tepat. Dengan PyTorch, kita bisa mulai bekerja pada masalah yang lebih nyata. Langkah kita selanjutnya dalam perjalanan membangun "Otak" VTuber adalah belajar cara:
1. Memuat dataset yang besar dan kompleks.
2. Memuat arsitektur model yang sudah ada (seperti Llama atau Mistral).
3. Melakukan **fine-tuning**: mengambil model yang sudah pintar itu dan melatihnya lagi sedikit pada data spesifik kita (transkrip VTuber, dll.) agar "kepribadiannya" berubah.

Jalankan kode `pytorch_xor.py` (jangan lupa `pip install torch` di dalam `.venv` milikmu jika belum ada). Rasakan betapa lebih sederhananya prosesnya. Ini adalah fondasi baru kita untuk membangun hal-hal yang jauh lebih besar.