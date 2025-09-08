import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
from datasets import load_dataset
import time

# --- LANGKAH PENTING: MENENTUKAN PERANGKAT ---
# Cek apakah GPU (CUDA) tersedia, jika tidak, gunakan CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Menggunakan perangkat: {device}")


# --- LANGKAH 1: MENYIAPKAN MODEL, TOKENIZER, DAN DATASET ---
MODEL_NAME = 'distilbert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# TAMBAH: Pindahkan model ke perangkat yang ditentukan (.to(device))
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(device)

raw_datasets = load_dataset("glue", "sst2")

# --- LANGKAH 2: PRA-PEMROSESAN DATA (TOKENISASI) ---
def tokenize_function(examples):
    return tokenizer(examples["sentence"], padding="max_length", truncation=True)

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(["sentence", "idx"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")

# --- LANGKAH 3: MEMBUAT DATALOADER ---
BATCH_SIZE = 16
train_dataloader = DataLoader(tokenized_datasets["train"], shuffle=True, batch_size=BATCH_SIZE)
eval_dataloader = DataLoader(tokenized_datasets["validation"], batch_size=BATCH_SIZE)

# --- LANGKAH 4: MENDEFINISIKAN OPTIMIZER DAN TRAINING LOOP ---
optimizer = AdamW(model.parameters(), lr=5e-5)
EPOCHS = 1

start_time = time.time()
model.train()
for epoch in range(EPOCHS):
    for i, batch in enumerate(train_dataloader):
        # TAMBAH: Pindahkan setiap batch data ke GPU saat training
        batch = {k: v.to(device) for k, v in batch.items()}
        
        outputs = model(**batch)
        loss = outputs.loss
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if (i + 1) % 200 == 0:
            print(f'Epoch [{epoch+1}/{EPOCHS}], Step [{i+1}/{len(train_dataloader)}], Loss: {loss.item():.4f}')

print(f"\nFine-tuning Selesai dalam {time.time() - start_time:.2f} detik")

# --- LANGKAH 5: EVALUASI DAN UJI COBA ---
kalimat_uji = [
    "Saya pikir film ini cukup bagus.",
    "Filmnya tidak jelek, tapi juga tidak bagus.",
    "Gila, gamenya seru abis!"
]

print("\n--- Menguji Model Setelah Fine-Tuning ---")
model.eval()
for kalimat in kalimat_uji:
    # TAMBAH: Pindahkan input tes ke GPU juga
    inputs = tokenizer(kalimat, return_tensors="pt").to(device)
    
    with torch.no_grad():
        logits = model(**inputs).logits
    
    predicted_class_id = logits.argmax().item()
    prediction = model.config.id2label[predicted_class_id]
    print(f"Kalimat: '{kalimat}' -> Prediksi: {prediction}")
