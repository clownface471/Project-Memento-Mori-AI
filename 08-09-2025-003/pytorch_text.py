import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

# --- LANGKAH 1: MENYIAPKAN DATA MENTAH & TOKENIZER ---

# Kita akan menggunakan tokenizer dari BERT, salah satu model bahasa paling populer.
# Tokenizer ini sudah dilatih pada data internet yang sangat besar.
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Dataset sederhana untuk klasifikasi sentimen
sentences = [
    "Saya sangat suka film ini!",
    "Ini adalah hari yang indah.",
    "Saya merasa luar biasa hari ini.",
    "Produk ini sangat buruk.",
    "Saya benci menunggu dalam antrian.",
    "Pengalaman yang sangat mengecewakan."
]
# Labels: 1 untuk Positif, 0 untuk Negatif
labels = [1, 1, 1, 0, 0, 0]


# --- LANGKAH 2: MEMBUAT CUSTOM DATASET UNTUK TEKS ---
# Ini adalah bagian terpenting. Kita membuat "buku resep" kita sendiri.
class SentimentDataset(Dataset):
    def __init__(self, sentences, labels, tokenizer, max_len):
        self.sentences = sentences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        label = self.labels[idx]

        # Di sinilah proses tokenisasi terjadi
        encoding = self.tokenizer.encode_plus(
            sentence,
            add_special_tokens=True, # Tambah token [CLS] dan [SEP]
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length', # Buat semua kalimat sama panjang
            truncation=True, # Potong jika lebih panjang
            return_attention_mask=True,
            return_tensors='pt', # Kembalikan sebagai PyTorch Tensor
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# --- LANGKAH 3: MEMBUAT DATALOADER & MEMBANGUN MODEL ---

MAX_LEN = 32 # Panjang maksimal kalimat setelah ditokenisasi
BATCH_SIZE = 2

# Buat instance dari Dataset dan DataLoader
train_dataset = SentimentDataset(sentences, labels, tokenizer, MAX_LEN)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)


# Arsitektur model yang sangat sederhana untuk teks
class SentimentClassifier(nn.Module):
    def __init__(self, vocab_size):
        super(SentimentClassifier, self).__init__()
        # Embedding mengubah ID token menjadi vektor yang punya makna
        self.embedding = nn.Embedding(vocab_size, 128)
        # Lapisan linear sederhana untuk klasifikasi
        self.fc = nn.Linear(128, 2) # Output 2 kelas (Positif/Negatif)

    def forward(self, input_ids, attention_mask):
        # Abaikan 'attention_mask' untuk model sederhana ini
        embedded = self.embedding(input_ids)
        # Ambil rata-rata dari embedding semua token dalam satu kalimat
        pooled = embedded.mean(dim=1)
        output = self.fc(pooled)
        return output

# Dapatkan ukuran kamus dari tokenizer
model = SentimentClassifier(tokenizer.vocab_size)

# --- LANGKAH 4: TRAINING LOOP (Sangat Mirip dengan MNIST!) ---

loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
EPOCHS = 20

model.train()
for epoch in range(EPOCHS):
    for batch in train_loader:
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']

        outputs = model(input_ids, attention_mask)
        loss = loss_function(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 5 == 0:
      print(f'Epoch [{epoch+1}/{EPOCHS}], Loss: {loss.item():.4f}')

# --- LANGKAH 5: UJI COBA ---
model.eval()
test_sentence = "Saya pikir film ini cukup bagus."
encoding = tokenizer.encode_plus(
    test_sentence,
    max_length=MAX_LEN,
    padding='max_length',
    truncation=True,
    return_tensors='pt'
)
input_ids = encoding['input_ids']
attention_mask = encoding['attention_mask']

with torch.no_grad():
    outputs = model(input_ids, attention_mask)
    _, prediction = torch.max(outputs, dim=1)
    sentiment = "Positif" if prediction.item() == 1 else "Negatif"
    print(f"\nKalimat tes: '{test_sentence}'")
    print(f"Prediksi sentimen: {sentiment}")
