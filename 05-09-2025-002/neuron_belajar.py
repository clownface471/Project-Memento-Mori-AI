import numpy as np

# Kita gunakan kembali Class Neuron dari modul sebelumnya,
# tapi dengan beberapa tambahan penting untuk proses belajar.

class Neuron:
    def __init__(self, jumlah_input):
        self.weights = np.random.rand(jumlah_input)
        self.bias = np.random.rand(1) # Bias juga kita buat acak sekarang

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # FUNGSI BARU: Turunan dari Sigmoid
    # Diperlukan untuk Kalkulus (Backpropagation)
    def _sigmoid_derivative(self, x):
        return self._sigmoid(x) * (1 - self._sigmoid(x))

    def feedforward(self, inputs):
        total = np.dot(inputs, self.weights) + self.bias
        return self._sigmoid(total)

    # FUNGSI PALING PENTING: Proses Training
    def train(self, training_inputs, training_outputs, learning_rate, epochs):
        print("Memulai Training...")
        for epoch in range(epochs):
            total_error = 0
            for inputs, correct_answer in zip(training_inputs, training_outputs):
                # Langkah 1: Tebak (Feedforward)
                predicted_output = self.feedforward(inputs)

                # Langkah 2: Ukur Seberapa Salah (Loss/Error)
                error = correct_answer - predicted_output
                total_error += abs(error)

                # Langkah 3 & 4: Cari Arah & Perbaiki (Backpropagation & Gradient Descent)
                # Ini adalah implementasi dari Kalkulus (aturan rantai/chain rule)
                adjustment = learning_rate * error * self._sigmoid_derivative(predicted_output)
                
                # Perbarui weights dan bias
                # --- INI BAGIAN YANG DIPERBAIKI ---
                # Kita menggunakan perkalian element-wise, bukan dot product.
                self.weights += inputs * adjustment
                self.bias += np.sum(adjustment)

            if (epoch % 1000) == 0:
                print(f"Epoch {epoch}, Total Error: {total_error[0]:.6f}")

        print("Training Selesai!")

# ==============================================================================
# BAGIAN EKSEKUSI UTAMA
# ==============================================================================
if __name__ == "__main__":
    # Buat dataset training
    # Input: [Tinggi (cm, dinormalisasi), Berat (kg, dinormalisasi)]
    # Normalisasi: (nilai - 150) / 50 untuk tinggi, (nilai - 50) / 50 untuk berat
    training_inputs = np.array([[0.2, 0.2],   # Tinggi 160, Berat 60
                                [0.4, 0.4],   # Tinggi 170, Berat 70
                                [0.6, 0.6],   # Tinggi 180, Berat 80
                                [-0.2, -0.2], # Tinggi 140, Berat 40
                                [-0.4, -0.4], # Tinggi 130, Berat 30
                                [-0.6, -0.6]  # Tinggi 120, Berat 20
                               ])

    # Output: [Jenis Kelamin] (0 = Wanita, 1 = Pria)
    training_outputs = np.array([[1, 1, 1, 0, 0, 0]]).T

    # 'Lahirkan' sebuah neuron yang siap menerima 2 input
    neuron = Neuron(2)

    print("Bobot Awal (Acak):", neuron.weights)
    print("Bias Awal (Acak):", neuron.bias)

    # Latih neuron!
    # learning_rate = 0.1 -> Seberapa besar langkah perbaikan
    # epochs = 10000 -> Berapa kali kita mengulang seluruh dataset
    neuron.train(training_inputs, training_outputs, 0.1, 10000)

    print("\nBobot Setelah Training:", neuron.weights)
    print("Bias Setelah Training:", neuron.bias)

    # --- UJI COBA ---
    # Buat data baru yang belum pernah dilihat neuron
    # Test 1: Tinggi 175cm (0.5), Berat 75kg (0.5) -> Seharusnya mendekati 1 (Pria)
    new_data_1 = np.array([0.5, 0.5])
    # Test 2: Tinggi 135cm (-0.3), Berat 35kg (-0.3) -> Seharusnya mendekati 0 (Wanita)
    new_data_2 = np.array([-0.3, -0.3])

    print(f"\nPrediksi untuk data baru [0.5, 0.5]: {neuron.feedforward(new_data_1)[0]:.4f}")
    print(f"Prediksi untuk data baru [-0.3, -0.3]: {neuron.feedforward(new_data_2)[0]:.4f}")

