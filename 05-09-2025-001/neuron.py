# Impor library NumPy, yang sangat penting untuk komputasi numerik di Python.
# Kita akan menggunakannya untuk operasi matematika pada array (vektor/matriks).
import numpy as np

# Ini adalah 'cetak biru' atau 'blueprint' untuk membuat sebuah Neuron.
# Di dalam pemrograman, ini disebut 'Class'.
class Neuron:
    """
    Setiap neuron yang kita buat akan memiliki 'bobot' (weights) dan 'bias'.
    Jumlah 'bobot' harus sama dengan jumlah input yang akan diterima neuron.
    """
    def __init__(self, jumlah_input):
        # Saat sebuah neuron 'dilahirkan' (diinisialisasi), kita berikan bobot acak.
        # Ini melambangkan bahwa neuron tersebut belum tahu apa-apa.
        self.weights = np.random.rand(jumlah_input)
        
        # Bias kita mulai dari 0. Bias ini seperti 'kecenderungan bawaan' neuron.
        self.bias = 0

    """
    Fungsi Aktivasi: Sigmoid
    Fungsinya adalah untuk 'menekan' hasil kalkulasi menjadi sebuah nilai
    antara 0 dan 1. Angka ini bisa kita interpretasikan sebagai probabilitas
    atau tingkat kepercayaan.
    - Output mendekati 0 berarti 'tidak aktif' / 'keputusan tidak'.
    - Output mendekati 1 berarti 'aktif' / 'keputusan ya'.
    """
    def _sigmoid(self, x):
        # Rumus matematika: 1 / (1 + e^(-x))
        return 1 / (1 + np.exp(-x))

    """
    Metode 'feedforward' adalah proses 'berpikir' dari neuron.
    Ia menerima data input dan menghasilkan sebuah output.
    """
    def feedforward(self, inputs):
        # Langkah 1: Kalkulasi total (weighted sum).
        # Ini adalah inti dari Aljabar Linear: Dot Product.
        # (input_1 * weight_1) + (input_2 * weight_2) + ... + bias
        total = np.dot(inputs, self.weights) + self.bias
        
        # Langkah 2: Masukkan hasil total ke fungsi aktivasi.
        # Ini adalah output akhir dari neuron.
        return self._sigmoid(total)

# ==============================================================================
# BAGIAN EKSEKUSI UTAMA
# Kode di bawah ini hanya akan berjalan jika file ini dieksekusi secara langsung.
# ==============================================================================
if __name__ == "__main__":
    print("Menciptakan dan Menguji Satu Neuron...")

    # Kita 'lahirkan' sebuah neuron yang siap menerima 3 input
    neuron_kopi = Neuron(3)

    # Untuk tujuan demonstrasi, kita tentukan bobot & bias secara manual
    # agar hasilnya bisa diprediksi dan dipelajari.
    # Bobot untuk input: [Hujan, Payung, Teman]
    neuron_kopi.weights = np.array([0.5, 0.8, 1.5]) 
    # Bias: -1.0 (kecenderungan bawaan untuk agak malas)
    neuron_kopi.bias = -1.0

    print(f"\nBobot Neuron diatur ke: {neuron_kopi.weights}")
    print(f"Bias Neuron diatur ke: {neuron_kopi.bias}")

    # Mari kita simulasikan sebuah skenario
    # Skenario: Tidak hujan (0), punya payung (1), teman ikut (1)
    inputs_skenario = np.array([0, 1, 1])
    print(f"\nInput Skenario: {inputs_skenario} (Tidak Hujan, Ada Payung, Teman Ikut)")

    # Minta neuron untuk 'berpikir' (melakukan feedforward)
    output = neuron_kopi.feedforward(inputs_skenario)

    # Tampilkan hasilnya
    print("\n--- Hasil Keputusan Neuron ---")
    print(f"Output (tingkat kepercayaan): {output:.4f}") # Dibulatkan 4 angka di belakang koma

    if output > 0.7:
        print("Interpretasi: Sangat mungkin untuk pergi ngopi!")
    elif output > 0.5:
        print("Interpretasi: Cenderung untuk pergi ngopi.")
    else:
        print("Interpretasi: Cenderung untuk tidak pergi (tinggal di rumah).")
