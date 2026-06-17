"""
Amaç:
    - Mini LibriSpeech veri setini indir ve temel özelliklerini analiz et
    - veri seti içinde bulunan ses dosyası, konuşmacı, bölüm ve transcript sayılarını ekrana yazdır

Veri seti: https://www.openslr.org/31/?utm_source=chatgpt.com
    - 16 kHz örnekleme frekansına sahip
    - Kısa İngilizce konuşma metinleri

Adımlar:
    1. Gerekli kütüphanelerin içeriye aktarılması
    2. Proje klasör yollarının tanımlanması
    3. Veri seti klasörünün kontrol edilmesi
    4. Veri seti bilgilerinin hesaplanması
    5. Veri seti bilgilerinin ekrana yazdırılması
    6. Ana çalışma akışının çalıştırılması
"""
# 1. Gerekli kütüphanelerin içeriye aktarılması
from pathlib import Path # dosya ve klasör yollarını yönetmek için kullanılır

# 2. Proje klasör yollarının tanımlanması
BASE_DIR = Path(__file__).resolve().parent # Bu python dosyasının bulunduğu klasörü proje ana klasörü olarak alır
DATA_DIR = BASE_DIR / "data" # tüm veri dosyaları için ana klasör
RAW_DATA_DIR = DATA_DIR / "raw" # ham veri setinin tutulacağı klasör
DATASET_DIR = RAW_DATA_DIR / "LibriSpeech" / "train-clean-5" # libri speech train clean 5 veri seti klasörü

# 3. Veri seti klasörünün kontrol edilmesi
def check_dataset_folder(): 
    """Veri seti klasörünün var olup olmadığını kontrol eder"""
    if not DATASET_DIR.exists(): # beklenen veri seti klasörü yoksa
        print(f"Veri seti klasörü bulunamadı: {DATASET_DIR}")
        return False
    
    return True

# 4. Veri seti bilgilerinin hesaplanması
def calculate_dataset_info():
    """Veri setiyle ilgili temel bilgileri hesaplar"""
    flac_files = list(DATA_DIR.rglob("*.flac")) # veri setinde ki tüm .flac ses dosyalarını listeye alır
    transcript_files = list(DATASET_DIR.rglob("*.trans.txt"))

    speaker_ids = {file.parts[-3] for file in flac_files} # dosya yolundan konuşmacı id değerlerini çıkart
    chapter_ids = {file.parts[-2] for file in flac_files} # dosya yolundan bölüm id değerlerinin çıkartılması

    dataset_info = {
        "dataset_path": DATASET_DIR, 
        "audio_file_count": len(flac_files), # toplam ses dosyası sayısı
        "speaker_count": len(speaker_ids), # toplam konuşmacı sayısı
        "chapter_count": len(chapter_ids), # toplam bölüm sayısını sözlüğe ekler
        "transcript_file_count": len(transcript_files) # toplam transcript dosyası sayısı
    }

    return dataset_info

# 5. Veri seti bilgilerinin ekrana yazdırılması
def show_dataset_info(dataset_info):
    """Hesaplanan veri seti bilgilerini ekrana yazdır"""
    print("Veri setiyle ilgili açıklama")
    print(f"Veri seti klasörü: {dataset_info["dataset_path"]}")
    print(f"Ses dosyası sayısı: {dataset_info["audio_file_count"]}")
    print(f"Konuşmacı sayısı: {dataset_info["speaker_count"]}")
    print(f"Bölüm sayısı: {dataset_info["chapter_count"]}")
    print(f"Transcript dosyası sayısı: {dataset_info["transcript_file_count"]}")

# 6. Ana çalışma akışının çalıştırılması
def main():
    """Ana çalışma klasörü"""

    is_dataset_ready = check_dataset_folder()

    if not is_dataset_ready:
        return # programı sonlandır
    
    dataset_info = calculate_dataset_info() # veri setiyle ilgili temel bilgileri hesaplar
    show_dataset_info(dataset_info) # hesaplanan veri seti bilgilerini ekrana yazdırır

if __name__ == "__main__":
    main()