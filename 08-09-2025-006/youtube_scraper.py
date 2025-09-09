import subprocess
import re
import os
from urllib.parse import urlparse, parse_qs

def extract_video_id(url_or_id):
    """
    Fungsi cerdas untuk mengekstrak ID video dari berbagai format URL YouTube.
    """
    if re.match(r'^[a-zA-Z0-9_-]{11}$', url_or_id):
        return url_or_id
    
    try:
        parsed_url = urlparse(url_or_id)
        if "youtube.com" in parsed_url.hostname:
            if parsed_url.path == '/watch':
                return parse_qs(parsed_url.query)['v'][0]
            if parsed_url.path.startswith(('/embed/', '/shorts/', '/live/')):
                return parsed_url.path.split('/')[2]
        elif "youtu.be" in parsed_url.hostname:
            return parsed_url.path[1:]
    except Exception:
        return None
    
    return None

def parse_vtt(vtt_content):
    """
    Fungsi untuk membersihkan konten file VTT dan hanya mengambil teksnya.
    """
    lines = vtt_content.strip().split('\n')
    text_lines = []
    for line in lines:
        if '-->' in line or line.strip().isdigit() or line.strip() == 'WEBVTT' or not line.strip():
            continue
        clean_line = re.sub(r'<[^>]+>', '', line)
        text_lines.append(clean_line.strip())
        
    return " ".join(text_lines)

def get_transcript_with_yt_dlp(video_id):
    """
    Fungsi utama yang menggunakan yt-dlp untuk mengunduh dan memproses transrip.
    """
    try:
        print(f"Menggunakan yt-dlp untuk video ID: {video_id}...")
        video_url = f"https://www.youtube.com/watch?v={video_id}"
        
        temp_vtt_filename = f"{video_id}" # Nama dasar, ekstensi akan ditambahkan oleh yt-dlp

        # PERUBAHAN DI SINI: Memberi nilai pada --impersonate
        command = [
            'yt-dlp',
            '--impersonate', 'chrome110', # Menyamar sebagai Chrome versi 110
            '--write-auto-sub',
            '--sub-lang', 'id,en',
            '--skip-download',
            '-o', temp_vtt_filename,
            video_url
        ]
        
        # Jalankan perintah di terminal
        # Menambahkan timeout untuk mencegah hang jika terjadi masalah jaringan
        subprocess.run(command, check=True, capture_output=True, text=True, timeout=120)
        
        actual_vtt_file = None
        for lang in ['id', 'en']:
            expected_filename = f"{video_id}.{lang}.vtt"
            if os.path.exists(expected_filename):
                actual_vtt_file = expected_filename
                break

        if not actual_vtt_file:
            # Jika tidak ada subtitle otomatis, coba subtitle manual
            print("Subtitle otomatis tidak ditemukan, mencoba subtitle manual...")
            command = [
                'yt-dlp',
                '--impersonate', 'chrome110',
                '--write-sub', # Ganti ke --write-sub untuk subtitle manual
                '--sub-lang', 'id,en',
                '--skip-download',
                '-o', temp_vtt_filename,
                video_url
            ]
            subprocess.run(command, check=True, capture_output=True, text=True, timeout=120)

            for lang in ['id', 'en']:
                expected_filename = f"{video_id}.{lang}.vtt"
                if os.path.exists(expected_filename):
                    actual_vtt_file = expected_filename
                    break
            
            if not actual_vtt_file:
                print(f"Error: Tidak ada file transkrip (otomatis atau manual) yang ditemukan.")
                return

        print(f"File transkrip '{actual_vtt_file}' berhasil diunduh. Memproses...")

        with open(actual_vtt_file, 'r', encoding='utf-8') as f:
            vtt_content = f.read()
        
        full_transcript = parse_vtt(vtt_content)
        
        final_filename = f"{video_id}_transcript.txt"
        with open(final_filename, 'w', encoding='utf-8') as f:
            f.write(full_transcript)
        
        print(f"Sukses! Transkrip telah disimpan sebagai '{final_filename}'")

        os.remove(actual_vtt_file)

    except subprocess.CalledProcessError as e:
        print(f"\n--- Error dari yt-dlp ---")
        print(f"yt-dlp gagal dengan pesan:")
        print(e.stderr)
        print(f"--------------------------")
        print("Ini biasanya berarti tidak ada subtitle yang tersedia atau permintaan diblokir.")
    except subprocess.TimeoutExpired:
        print("Error: Proses yt-dlp memakan waktu terlalu lama (timeout). Coba lagi atau periksa koneksi.")
    except Exception as e:
        print(f"Terjadi error yang tidak terduga: {e}")


# --- BAGIAN UTAMA PROGRAM ---
if __name__ == "__main__":
    print("--- Pustakawan Digital Mori v2.2 (Penyamaran Sempurna) ---")
    user_input = input("Masukkan URL atau ID Video YouTube target: ")
    
    if user_input:
        video_id = extract_video_id(user_input)
        if video_id:
            get_transcript_with_yt_dlp(video_id)
        else:
            print("Error: URL atau ID video tidak valid.")
    else:
        print("Tidak ada input yang dimasukkan. Program berhenti.")

