# Impor library yang kita butuhkan
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import NoTranscriptFound, TranscriptsDisabled
from urllib.parse import urlparse, parse_qs
import re

def extract_video_id(url_or_id):
    """
    Fungsi cerdas untuk mengekstrak ID video dari berbagai format URL YouTube.
    Jika input sudah berupa ID, ia akan mengembalikannya langsung.
    """
    # Pola regex untuk mencocokkan ID video YouTube yang valid (11 karakter)
    if re.match(r'^[a-zA-Z0-9_-]{11}$', url_or_id):
        return url_or_id
    
    # Jika input adalah URL, coba parsing
    try:
        parsed_url = urlparse(url_or_id)
        if "youtube.com" in parsed_url.hostname:
            # Untuk URL standar seperti /watch?v=...
            if parsed_url.path == '/watch':
                return parse_qs(parsed_url.query)['v'][0]
            # Untuk URL pendek seperti /shorts/... atau /...
            if parsed_url.path.startswith(('/embed/', '/shorts/', '/live/')):
                return parsed_url.path.split('/')[2]
        elif "youtu.be" in parsed_url.hostname:
            # Untuk URL youtu.be/...
            return parsed_url.path[1:]
    except Exception:
        return None # Gagal mengekstrak ID
    
    return None

def get_and_save_transcript(video_id):
    """
    Fungsi untuk mengambil transkrip dari YouTube dan menyimpannya ke file teks.
    """
    try:
        print(f"Mencoba mengambil transkrip untuk video ID: {video_id}...")
        
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=['id', 'en'])

        print("Transkrip berhasil ditemukan. Membersihkan teks...")
        
        full_transcript = " ".join([segment['text'] for segment in transcript_list])
        
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
    
    # Minta pengguna untuk memasukkan URL atau ID video
    user_input = input("Masukkan URL atau ID Video YouTube target: ")
    
    if user_input:
        # Ekstrak ID video dari input pengguna
        video_id = extract_video_id(user_input)
        
        if video_id:
            get_and_save_transcript(video_id)
        else:
            print("Error: URL atau ID video tidak valid.")
    else:
        print("Tidak ada input yang dimasukkan. Program berhenti.")

