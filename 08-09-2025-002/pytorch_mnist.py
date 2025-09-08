import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time

# --- LANGKAH 1: MENDEFINISIKAN HIPERPARAMETER ---
# Menempatkan semua pengaturan di atas membuat kode lebih rapi.
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.001

# --- LANGKAH 2: MEMUAT DAN MEMPERSIAPKAN DATA ---
# 'transforms' adalah serangkaian operasi preprocessing untuk data kita.
transform = transforms.Compose([
    transforms.ToTensor(), # Mengubah gambar menjadi PyTorch Tensor
    transforms.Normalize((0.5,), (0.5,)) # Menormalisasi nilai piksel ke rentang [-1, 1]
])

# Unduh dataset training MNIST
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
# Unduh dataset testing MNIST
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Buat DataLoader untuk training dan testing
# DataLoader akan mengurus batching dan shuffling.
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# --- LANGKAH 3: MEMBANGUN ARSITEKTUR JARINGAN ---
# Jaringan ini dirancang untuk data gambar yang sudah 'diratakan' (flattened).
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.network_stack = nn.Sequential(
            nn.Linear(28*28, 512), # Input 784 (gambar 28x28)
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10) # Output 10 (untuk 10 kelas angka 0-9)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.network_stack(x)
        return logits

model = NeuralNetwork()

# --- LANGKAH 4: MENDEFINISIKAN LOSS & OPTIMIZER ---
# Menggunakan CrossEntropyLoss, yang cocok untuk klasifikasi multi-kelas.
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# --- LANGKAH 5: SIKLUS PELATIHAN ---
start_time = time.time()
for epoch in range(EPOCHS):
    # Iterasi melalui DataLoader untuk setiap batch data
    for batch_idx, (images, labels) in enumerate(train_loader):
        # Forward pass
        outputs = model(images)
        loss = loss_function(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (batch_idx + 1) % 200 == 0:
            print(f'Epoch [{epoch+1}/{EPOCHS}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

print(f"\nTraining Selesai dalam {time.time() - start_time:.2f} detik")

# --- LANGKAH 6: EVALUASI MODEL ---
# Pindahkan model ke mode evaluasi
model.eval() 
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        outputs = model(images)
        # Dapatkan kelas yang diprediksi dari output dengan probabilitas tertinggi
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Akurasi model pada 10,000 gambar tes: {accuracy:.2f} %')

