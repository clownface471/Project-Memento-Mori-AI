import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- LANGKAH 1: MEMUAT MODEL & TOKENIZER YANG SUDAH DI-FINE-TUNE ---
# Pastikan path ini sesuai dengan tempat Anda menyimpan model hasil fine-tuning.
MODEL_PATH = "./mori_brain_indo_final_v3"
print(f"--- Memuat 'Otak' Mori dari: {MODEL_PATH} ---")

# Gunakan try-except untuk error handling yang lebih baik
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
    print("Model dan tokenizer berhasil dimuat.")
except OSError:
    print(f"ERROR: Tidak dapat menemukan model di path '{MODEL_PATH}'.")
    print("Pastikan Anda sudah menjalankan skrip fine-tuning dan modelnya tersimpan di direktori yang benar.")
    exit()

# Pindahkan model ke GPU jika tersedia untuk inferensi yang lebih cepat
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval() # Pindahkan model ke mode evaluasi

# --- LANGKAH 2: MENYIAPKAN PROMPT UNTUK PENGUJIAN ---
# Kita akan menguji Mori dengan berbagai jenis prompt untuk melihat reaksinya.
test_prompts = [
    "Halo semua, selamat datang di stream kali ini kita akan",
    "Menurut kamu, game Lisa the Painful itu tentang apa?",
    "Apa yang harus aku lakukan kalau ketemu hantu?",
    "Ceritakan sebuah lelucon yang absurd.",
    "Why do they call it oven when you of in the cold food of out hot eat the food?", # Tes dengan input Bahasa Inggris & kacau
]

# --- LANGKAH 3: PENGUJIAN GENERASI TEKS ---
print("\n--- Memulai Sesi Tanya Jawab dengan Mori v0.1 ---")

for prompt in test_prompts:
    print("\n==================================================")
    print(f"PROMPT: {prompt}")
    print("==================================================")
    
    # Ubah prompt menjadi format yang dimengerti model (token)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Lakukan proses generasi teks
    with torch.no_grad():
        # Kita minta 3 variasi jawaban untuk melihat kreativitasnya
        outputs = model.generate(
            **inputs, 
            max_new_tokens=100, # Batasi panjang token baru yang dihasilkan
            num_return_sequences=3, # Hasilkan 3 output yang berbeda
            do_sample=True, # Aktifkan sampling untuk hasil yang lebih kreatif
            top_k=50,
            top_p=0.95,
            temperature=0.7, # PERUBAHAN: Turunkan temperature dari 0.8 menjadi 0.7
            repetition_penalty=1.2, # PERUBAHAN: Tambahkan penalti untuk pengulangan
            pad_token_id=tokenizer.eos_token_id # Penting untuk open-end generation
        )

    # Tampilkan hasil
    print("RESPONS MORI:")
    for i, output in enumerate(outputs):
        generated_text = tokenizer.decode(output, skip_special_tokens=True)
        print(f"--- Jawaban #{i+1} ---")
        print(generated_text)
        print("-" * 20)

print("\n--- Sesi Pengujian Selesai ---")
