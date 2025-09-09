import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset
import os
import glob

# --- [BAGIAN BARU] LANGKAH 0: PERSIAPAN CORPUS OTOMATIS ---
TRANSCRIPTS_FOLDER = "transcripts"
CORPUS_FILE = "mori_corpus.txt"

print("--- Memulai Persiapan Data ---")
# Cek apakah folder transcripts ada
if not os.path.isdir(TRANSCRIPTS_FOLDER):
    print(f"Error: Folder '{TRANSCRIPTS_FOLDER}' tidak ditemukan.")
    print("Silakan buat folder 'transcripts' dan unggah file .txt ke dalamnya.")
else:
    # Gabungkan semua file .txt dari folder transcripts menjadi satu corpus
    search_path = os.path.join(TRANSCRIPTS_FOLDER, '*.txt')
    transcript_files = glob.glob(search_path)
    
    if not transcript_files:
        print(f"Error: Tidak ada file .txt yang ditemukan di dalam folder '{TRANSCRIPTS_FOLDER}'.")
    else:
        print(f"Menemukan {len(transcript_files)} file transkrip. Menggabungkan menjadi '{CORPUS_FILE}'...")
        with open(CORPUS_FILE, 'w', encoding='utf-8') as outfile:
            for filename in transcript_files:
                with open(filename, 'r', encoding='utf-8') as infile:
                    outfile.write(infile.read() + "\n\n") # Tambahkan baris baru untuk pemisah
        print("Corpus berhasil dibuat.")


# --- LANGKAH 1: PERSIAPAN MODEL DAN TOKENIZER ---
MODEL_NAME = "cahya/gpt2-small-indonesian-522M"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
    model.resize_token_embeddings(len(tokenizer))


# --- LANGKAH 2: MEMUAT DAN MEMPERSIAPKAN CORPUS KITA ---
print("\n--- Memuat dan Memproses Corpus ---")
dataset = load_dataset('text', data_files={'train': CORPUS_FILE})

# ... sisa kode dari sini ke bawah tetap sama persis ...
# (Fungsi tokenize_function, lm_dataset, TrainingArguments, Trainer, dll.)

def tokenize_function(examples):
    block_size = 128
    
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    
    total_length = (total_length // block_size) * block_size
    
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

tokenized_dataset = dataset.map(
    lambda examples: tokenizer(examples['text'], truncation=True, max_length=1024),
    batched=True, 
    remove_columns=["text"]
)

lm_dataset = tokenized_dataset.map(
    tokenize_function,
    batched=True,
)

print(f"Dataset berhasil diproses menjadi {len(lm_dataset['train'])} blok data.")

# --- LANGKAH 3: FINE-TUNING MENGGUNAKAN TRAINER API ---
training_args = TrainingArguments(
    output_dir="./mori_brain_indo_v3",
    overwrite_output_dir=True,
    num_train_epochs=5,
    per_device_train_batch_size=4,
    save_steps=10_000,
    save_total_limit=2,
    prediction_loss_only=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_dataset["train"],
)

print("\n--- Memulai proses fine-tuning 'Otak' Mori ---")
trainer.train()
print("Fine-tuning selesai!")

trainer.save_model("./mori_brain_indo_final_v3")
tokenizer.save_pretrained("./mori_brain_indo_final_v3")

# --- LANGKAH 4: MENGHASILKAN TEKS DARI "OTAK" MORI ---
print("\n--- Tes Generasi Teks ---")
model.eval()
prompt = "Halo semua, selamat datang di stream kali ini kita akan"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs, 
        max_length=100,
        num_return_sequences=1,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.9
    )

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Prompt: {prompt}")
print(f"Hasil Generasi Mori: {generated_text}")

