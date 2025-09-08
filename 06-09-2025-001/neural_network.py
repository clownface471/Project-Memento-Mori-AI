import numpy as np

# Kita tidak lagi butuh class Neuron terpisah.
# Logikanya akan kita bangun langsung di dalam class Jaringan kita.

class OurNeuralNetwork:
    """
    Sebuah jaringan saraf tiruan dengan:
      - 2 input
      - 1 hidden layer dengan 2 neuron (h1, h2)
      - 1 output layer dengan 1 neuron (o1)
    """
    def __init__(self):
        # Inisialisasi bobot (weights) dan bias secara acak
        # Bentuk (shape) dari matriks bobot sangat penting!
        
        # Bobot dari input layer ke hidden layer
        # Bentuk: (jumlah input, jumlah neuron di hidden layer) -> (2, 2)
        self.weights_h = np.random.normal(size=(2, 2))
        
        # Bobot dari hidden layer ke output layer
        # Bentuk: (jumlah neuron di hidden layer, jumlah neuron di output) -> (2, 1)
        self.weights_o = np.random.normal(size=(2, 1))

        # Bias untuk setiap neuron di hidden layer dan output layer
        self.bias_h = np.random.normal(size=(1, 2))
        self.bias_o = np.random.normal(size=(1, 1))

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _sigmoid_derivative(self, x):
        return x * (1 - x)

    def feedforward(self, x):
        # Proses dari input ke hidden layer
        hidden_input = np.dot(x, self.weights_h) + self.bias_h
        hidden_output = self._sigmoid(hidden_input)

        # Proses dari hidden ke output layer
        final_input = np.dot(hidden_output, self.weights_o) + self.bias_o
        final_output = self._sigmoid(final_input)
        
        return final_output

    def train(self, data, all_y_trues, learning_rate, epochs):
        for epoch in range(epochs):
            # --- Feedforward (sama seperti di atas, tapi kita simpan hasilnya) ---
            hidden_input = np.dot(data, self.weights_h) + self.bias_h
            hidden_output = self._sigmoid(hidden_input)

            final_input = np.dot(hidden_output, self.weights_o) + self.bias_o
            y_pred = self._sigmoid(final_input)

            # --- Backpropagation (bagian yang lebih kompleks) ---
            # 1. Hitung error di output layer
            output_error = all_y_trues - y_pred
            output_delta = output_error * self._sigmoid_derivative(y_pred)

            # 2. Hitung error di hidden layer (propagasi mundur)
            hidden_error = np.dot(output_delta, self.weights_o.T)
            hidden_delta = hidden_error * self._sigmoid_derivative(hidden_output)

            # --- Update bobot dan bias ---
            # Update bobot: hidden -> output
            self.weights_o += np.dot(hidden_output.T, output_delta) * learning_rate
            self.bias_o += np.sum(output_delta, axis=0, keepdims=True) * learning_rate
            
            # Update bobot: input -> hidden
            self.weights_h += np.dot(data.T, hidden_delta) * learning_rate
            self.bias_h += np.sum(hidden_delta, axis=0, keepdims=True) * learning_rate
            
            if epoch % 1000 == 0:
                loss = np.mean(np.abs(output_error))
                print(f"Epoch {epoch} Loss: {loss:.6f}")

# ==============================================================================
# BAGIAN EKSEKUSI UTAMA
# ==============================================================================
if __name__ == "__main__":
    # Dataset untuk masalah XOR
    data = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    all_y_trues = np.array([
        [0],
        [1],
        [1],
        [0]
    ])

    # Inisialisasi Jaringan
    network = OurNeuralNetwork()

    # Latih Jaringan
    network.train(data, all_y_trues, learning_rate=0.1, epochs=10000)

    # --- UJI COBA ---
    print("\n--- Testing ---")
    print(f"Prediksi untuk [0, 0]: {network.feedforward(np.array([0, 0]))[0][0]:.4f} (Seharusnya: 0)")
    print(f"Prediksi untuk [0, 1]: {network.feedforward(np.array([0, 1]))[0][0]:.4f} (Seharusnya: 1)")
    print(f"Prediksi untuk [1, 0]: {network.feedforward(np.array([1, 0]))[0][0]:.4f} (Seharusnya: 1)")
    print(f"Prediksi untuk [1, 1]: {network.feedforward(np.array([1, 1]))[0][0]:.4f} (Seharusnya: 0)")
