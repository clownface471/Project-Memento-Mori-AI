# Impor library yang kita butuhkan
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import NoTranscriptFound, TranscriptsDisabled

def get_and_save_transcript(video_id):
    """
    Fungsi untuk mengambil transkrip dari YouTube dan menyimpannya ke file teks.
    """
    try:
        print(f"Mencoba mengambil transkrip untuk video ID: {video_id}...")
        
        # Langkah 1: Panggil API untuk mendapatkan transkrip
        # 'languages=['id', 'en']' berarti ia akan mencoba mencari transkrip B. Indonesia dulu,
        # jika tidak ada, ia akan mencari transkrip B. Inggris.
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=['id', 'en'])

        print("Transkrip berhasil ditemukan. Membersihkan teks...")
        
        # Langkah 2: Proses transkrip untuk mendapatkan teks bersih
        # Kita hanya mengambil bagian 'text' dari setiap segmen dalam transkrip
        full_transcript = " ".join([segment['text'] for segment in transcript_list])
        
        # Langkah 3: Simpan teks bersih ke dalam file
        file_name = f"{video_id}_transcript.txt"
        with open(file_name, "w", encoding="utf-8") as f:
            f.write(full_transcript)
            
        print(f"Sukses! Transkrip telah disimpan sebagai '{file_name}'")

    except NoTranscriptFound:
        print(f"Error: Tidak ada transkrip yang ditemukan untuk video ID {video_id}. Coba video lain.")
    except TranscriptsDisabled:
        print(f"Error: Transkrip dinonaktifkan untuk video ID {video_id}.")
    except Exception as e:
        print(f"Terjadi error yang tidak terduga: {e}")


# --- BAGIAN UTAMA PROGRAM ---
if __name__ == "__main__":
    print("--- Alat Pengunduh Transkrip YouTube untuk Proyek Mori ---")
    
    # Minta pengguna untuk memasukkan ID video YouTube
    # Contoh ID: dari URL https://www.youtube.com/watch?v=dQw4w9WgXcQ, ID-nya adalah dQw4w9WgXcQ
    target_video_id = input("Masukkan ID Video YouTube target: ")
    
    if target_video_id:
        get_and_save_transcript(target_video_id)
    else:
        print("Tidak ada ID video yang dimasukkan. Program berhenti.")
