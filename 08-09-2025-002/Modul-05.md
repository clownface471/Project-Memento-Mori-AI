# Modul 5: Alur Kerja Profesional - Datasets & DataLoaders

Kita telah memasuki dunia *machine learning* yang sesungguhnya. Di modul ini, kita belajar alur kerja standar untuk menangani dataset yang besar dan kompleks, sebuah skill yang mutlak diperlukan untuk proyek VTuber AI kita.

### 1. Konsep Inti: Mengelola Skala

* **Masalah:** Data di dunia nyata (gambar, teks, audio) terlalu besar untuk dimuat ke memori sekaligus. Kita butuh cara yang efisien untuk memuatnya sedikit demi sedikit.
* **Solusi PyTorch:**
    * **`Dataset`:** Objek yang merepresentasikan keseluruhan data kita dan tahu cara mengambil satu sampel data individual (`__getitem__`).
    * **`DataLoader`:** Sebuah *iterator* yang membungkus `Dataset`. Ia secara otomatis mengambil data dalam **kelompok (*batch*)**, mengacaknya jika perlu, dan bahkan bisa menggunakan beberapa proses CPU untuk mempercepat pemuatan data.

### 2. Pra-pemrosesan Data dengan `transforms`

Data mentah jarang sekali bisa langsung dimasukkan ke model. Ia perlu "dibersihkan" dan diubah formatnya terlebih dahulu. Di PyTorch, ini ditangani oleh `transforms`. Dalam kode kita:

* **`transforms.ToTensor()`:** Mengubah gambar dari format PIL (Python Imaging Library) atau NumPy menjadi format `torch.Tensor`.
* **`transforms.Normalize((0.5,), (0.5,))`:** Menyesuaikan rentang nilai piksel dari [0, 1] menjadi [-1, 1]. Normalisasi membantu model belajar lebih cepat dan stabil.

### 3. Anatomi Kode: `pytorch_mnist.py`

* **Pemuatan Data:** Perhatikan bagaimana kita tidak lagi menulis data secara manual. Kita menggunakan `datasets.MNIST` untuk mengunduh dan membuat `Dataset` secara otomatis. Lalu kita membungkusnya dengan `DataLoader`.
* **Arsitektur Model:** Input model kita sekarang `28*28 = 784`, sesuai dengan jumlah piksel pada gambar MNIST yang sudah "diratakan". Outputnya adalah `10`, untuk setiap kemungkinan kelas angka (0-9).
* **Loss Function Baru: `nn.CrossEntropyLoss`:** Untuk masalah klasifikasi dengan lebih dari dua pilihan (multi-kelas), `CrossEntropyLoss` adalah pilihan standarnya. Ia secara internal menggabungkan aktivasi `Softmax` dan `Negative Log-Likelihood Loss`, menjadikannya sangat efisien dan stabil.
* **Training Loop Baru:** Perhatikan perubahan besarnya. Kita tidak lagi beriterasi berdasarkan `epoch` saja, tapi kita memiliki *loop* di dalam *loop*:
    ```python
    for epoch in range(EPOCHS):
        for images, labels in train_loader:
            # ... proses training untuk satu batch ...
    ```
    Ini adalah pola standar yang akan kamu lihat di hampir semua kode training PyTorch.
* **Mode Evaluasi (`model.eval()`):** Saat kita selesai training dan ingin menguji model, penting untuk memindahkannya ke mode evaluasi. Ini akan menonaktifkan beberapa perilaku spesifik training (seperti *dropout*) dan memastikan hasil evaluasi kita konsisten.

### 4. Apa Selanjutnya?

Kamu baru saja berhasil membangun, melatih, dan mengevaluasi sebuah **pengklasifikasi gambar** dari awal hingga akhir menggunakan alur kerja profesional. Ini adalah lompatan besar.

Langkah kita selanjutnya adalah mengambil alur kerja yang sama persis (`Dataset`, `DataLoader`, `training loop`) dan menerapkannya pada **data teks**. Kita akan mengganti `transforms` untuk gambar dengan proses yang disebut **tokenisasi** untuk teks. Setelah kita menguasai itu, kita akan siap untuk mulai mengerjakan "Otak" VTuber kita dengan melakukan *fine-tuning* pada model bahasa yang sudah ada.