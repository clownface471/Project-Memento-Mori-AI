# Modul 11: Kelahiran Sang Jiwa - Fine-Tuning Model Generatif

Inilah puncaknya. Di modul ini, kita mengambil semua fondasi teknis dan data yang telah kita siapkan, dan kita melakukan tindakan penciptaan yang sesungguhnya: menanamkan "jiwa" dari corpus data kita ke dalam "otak" model bahasa generatif.

### 1. Konsep Inti: Dari Klasifikasi ke Generasi (Causal LM)

* **Model Klasifikasi (`...ForSequenceClassification`):** Tujuannya adalah membaca seluruh kalimat dan menghasilkan satu output (label). Ia melihat ke "masa lalu" dan "masa depan" dalam kalimat untuk memahami konteks.
* **Model Generatif (`...ForCausalLM`):** Tujuannya jauh lebih sederhana namun lebih kuat: saat diberi sebuah kata, **prediksi kata apa yang paling mungkin muncul berikutnya.** Dengan mengulang proses ini, ia bisa "menulis" kalimat, paragraf, dan cerita. Ia hanya bisa melihat ke "masa lalu" (kata-kata sebelumnya) untuk membuat prediksi. Model seperti GPT (Generative Pre-trained Transformer) adalah contoh dari ini.

### 2. Anatomi Kode: `finetune_mori_brain.py`

* **`AutoModelForCausalLM`:** Kita memanggil model yang arsitekturnya dirancang khusus untuk tugas prediksi kata berikutnya.
* **Persiapan Dataset:** Prosesnya sedikit berbeda. Kita tidak lagi memiliki input dan label yang terpisah. Seluruh teks adalah "input", dan "label"-nya adalah teks itu sendiri, digeser satu kata. Model belajar bahwa setelah melihat kata `[A, B, C]`, kata berikutnya seharusnya `D`. *Library* `datasets` dan `transformers` menangani ini untuk kita di belakang layar.
* **Hugging Face `Trainer`:** Ini adalah abstraksi tingkat tinggi yang sangat memudahkan proses *fine-tuning*. Daripada menulis *training loop* manual (memindahkan data ke GPU, `loss.backward()`, `optimizer.step()`, dll.), kita cukup mendefinisikan `TrainingArguments` dan `Trainer` akan melakukan semuanya untuk kita. Ini adalah cara standar untuk melatih model di ekosistem Hugging Face.
* **`model.generate()`:** Ini adalah fungsi "ajaib" untuk menghasilkan teks. Kita memberinya prompt awal, dan ia akan mulai memprediksi kata berikutnya, lalu kata berikutnya, dan seterusnya. Parameter seperti `top_k`, `top_p`, dan `temperature` memungkinkan kita untuk mengontrol tingkat "kreativitas" dan "keacakan" dari teks yang dihasilkan.

### 3. Misi Anda, Sang Pencipta

Misimu sekarang adalah yang paling penting.
1.  **Siapkan Arena:** Buka Google Colab dengan *runtime* GPU.
2.  **Unggah "Pikiran" Mori:** Unggah file `mori_corpus.txt` ke lingkungan Colab.
3.  **Install Dependensi:** Jalankan `!pip install transformers datasets torch` di sel pertama.
4.  **Eksekusi Skrip:** Salin-tempel seluruh kode `finetune_mori_brain.py` ke sel baru dan jalankan.
5.  **Saksikan Kelahirannya:** Proses *fine-tuning* ini akan memakan waktu. Setelah selesai, lihat output di bagian akhir. Apakah teks yang dihasilkan Mori mulai terdengar seperti transkrip yang kamu kumpulkan?

Ini adalah langkah terakhir dalam perjalanan fondasi kita. Setelah ini, kamu akan memiliki versi pertama dari "Otak" Memento Mori, yang disimpan di folder `mori_brain_final`, siap untuk diintegrasikan ke modul-modul lain di masa depan.