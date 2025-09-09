import os
import glob

def create_corpus_from_transcripts(folder_path, output_file):
    """
    Menggabungkan semua file .txt dari sebuah folder menjadi satu file corpus.
    """
    try:
        # Buat path pencarian untuk semua file .txt di dalam folder
        search_path = os.path.join(folder_path, '*.txt')
        
        # Temukan semua file yang cocok dengan pola
        transcript_files = glob.glob(search_path)
        
        if not transcript_files:
            print(f"Error: Tidak ada file .txt yang ditemukan di folder '{folder_path}'")
            return

        print(f"Menemukan {len(transcript_files)} file transkrip. Memulai proses penggabungan...")

        # Buka file output untuk ditulis
        with open(output_file, 'w', encoding='utf-8') as outfile:
            for filename in transcript_files:
                # Buka setiap file transkrip untuk dibaca
                with open(filename, 'r', encoding='utf-8') as infile:
                    # Baca seluruh konten dan tulis ke file output
                    # Menambahkan spasi di akhir untuk memisahkan konten antar file
                    outfile.write(infile.read() + " ")
        
        print(f"Sukses! Seluruh transkrip telah digabungkan ke dalam file '{output_file}'")

    except FileNotFoundError:
        print(f"Error: Folder '{folder_path}' tidak ditemukan. Pastikan path-nya benar.")
    except Exception as e:
        print(f"Terjadi error yang tidak terduga: {e}")


# --- BAGIAN UTAMA PROGRAM ---
if __name__ == "__main__":
    print("--- Alat Persiapan Data untuk Proyek Mori ---")
    
    # Minta pengguna untuk memasukkan path ke folder berisi transkrip
    # Contoh: C:\Users\User\Downloads\AI-Learning\transcripts
    input_folder = input("Masukkan path ke folder berisi file-file transkrip .txt: ")
    
    if input_folder:
        output_corpus_file = "mori_corpus.txt"
        create_corpus_from_transcripts(input_folder, output_corpus_file)
    else:
        print("Tidak ada path folder yang dimasukkan. Program berhenti.")
