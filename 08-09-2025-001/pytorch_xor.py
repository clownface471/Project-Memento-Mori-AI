import torch
import torch.nn as nn

# --- LANGKAH 1: MENYIAPKAN DATA (Tetap Sama) ---
X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)


# --- LANGKAH 2: MEMBANGUN ARSITEKTUR JARINGAN (Di-upgrade) ---
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(2, 4)
        # GANTI: Menggunakan ReLU untuk hidden layer untuk menghindari vanishing gradient
        self.activation = nn.ReLU() 
        self.layer2 = nn.Linear(4, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        # HAPUS: Aktivasi Sigmoid terakhir dihapus karena akan ditangani oleh loss function
        return x

model = NeuralNetwork()


# --- LANGKAH 3: MENDEFINISIKAN LOSS & OPTIMIZER (Di-upgrade) ---
# GANTI: Menggunakan BCEWithLogitsLoss yang lebih cocok untuk klasifikasi biner
loss_function = nn.BCEWithLogitsLoss()

# GANTI: Menggunakan Adam, optimizer yang lebih modern dan seringkali lebih baik
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)


# --- LANGKAH 4: SIKLUS PELATIHAN (Tetap Sama Secara Logika) ---
epochs = 10000

for epoch in range(epochs):
    optimizer.zero_grad()
    
    # Forward pass menghasilkan 'logits' (output mentah sebelum aktivasi)
    logits = model(X)
    
    # Loss function membandingkan logits dengan jawaban benar
    loss = loss_function(logits, y)

    loss.backward()
    optimizer.step()

    if epoch % 1000 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item():.4f}')


# --- LANGKAH 5: UJI COBA MODEL YANG SUDAH DILATIH (Sedikit Disesuaikan) ---
print("\n--- Testing ---")
with torch.no_grad():
    for i, data_point in enumerate(X):
        # Dapatkan logits dari model
        logit = model(data_point)
        # Lewatkan logits melalui Sigmoid secara manual untuk mendapatkan probabilitas
        prediction = torch.sigmoid(logit)
        print(f"Prediksi untuk {data_point.tolist()}: {prediction.item():.4f} (Seharusnya: {y[i].item()})")

