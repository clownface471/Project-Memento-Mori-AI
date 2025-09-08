from transformers import pipeline

# --- LANGKAH 1: MEMBUAT PIPELINE ---
# Perintah ini akan mengunduh model dan tokenizer yang sudah di-fine-tune
# untuk analisis sentimen, lalu membungkusnya dalam objek yang mudah digunakan.
# (Model default-nya mungkin lebih baik dalam B. Inggris, tapi cukup pintar untuk B. Indonesia sederhana)
sentiment_pipeline = pipeline("sentiment-analysis")

# --- LANGKAH 2: MENYIAPKAN KALIMAT UJI COBA ---
kalimat_uji = [
    # Kalimat dari dataset kita sebelumnya
    "Saya sangat suka film ini!",
    "Produk ini sangat buruk.",
    "Saya benci menunggu dalam antrian.",
    "Ini adalah hari yang indah.",
    # Kalimat tes kita
    "Saya pikir film ini cukup bagus.",
    # Kalimat baru yang ambigu
    "Filmnya tidak jelek, tapi juga tidak bagus.",
    # Kalimat dengan gaya bahasa unik
    "Gila, gamenya seru abis!"
]

# --- LANGKAH 3: MENJALANKAN PREDIKSI ---
print("--- Menjalankan Prediksi dengan Model Pre-trained ---")
hasil = sentiment_pipeline(kalimat_uji)

for kalimat, sentimen in zip(kalimat_uji, hasil):
    label = sentimen['label']
    skor = sentimen['score']
    print(f"Kalimat: '{kalimat}'")
    print(f" -> Prediksi: {label} (Skor: {skor:.4f})\n")
