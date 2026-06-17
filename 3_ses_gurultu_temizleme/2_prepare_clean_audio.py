"""
Amaç:
    - .flac uzantılı konuşma seslerini .wav formatına dönüştür
    - Sesleri 16 kHz, mono ve normalize edilmiş hale getirerek gürültü ekleme adımı için hazırla

Veri seti: train-clean-5 temiz konuşma sesleri

Adımlar:
    1. Gerekli kütüphanelerin içeriye aktarılması
    2. Proje klasör yollarının ve ses ayarlarının tanımlanması
    3. Gerekli klasörlerin oluşturulması
    4. Temiz ses dosyalarının listelenmesi
    5. Tek bir ses dosyasını hazırlayan fonksiyon tanımla
    6. Tüm temiz ses dosyalarını .wav formatına dönüştür
    7. Ana çalışma akışını oluştur    
"""
# 1. Gerekli kütüphanelerin içeriye aktarılması
from pathlib import Path 
import numpy as np 
import librosa # ses dosyalarını okumak ve örnekleme frekansını ayarlamak
import soundfile as sf # .wav formatında kaydetmek için
from tqdm import tqdm # işlem ilerlemesini görsel olarak göstermek için

# 2. Proje klasör yollarının ve ses ayarlarının tanımlanması
BASE_DIR = Path(__file__).resolve().parent # Bu python dosyasının bulunduğu klasörü proje ana klasörü olarak alır
DATA_DIR = BASE_DIR / "data" # tüm veri dosyaları için ana klasör
RAW_DATA_DIR = DATA_DIR / "raw" # ham veri setinin tutulacağı klasör
DATASET_DIR = RAW_DATA_DIR / "LibriSpeech" / "train-clean-5" # libri speech train clean 5 veri seti klasörü
CLEAN_WAV_DIR = DATA_DIR / "clean_wav" # hazırlanan temiz wav dosyalarının kaydedileceği klasör

SAMPLE_RATE = 16000 # seslerin 16 kHz örnekleme frekansı ile hazırlanmasını sağlar
MAX_FILES = None # tüm dosyaları kullanmak için None, deneme için örneğin 100 yapabilirsiniz

# 3. Gerekli klasörlerin oluşturulması
CLEAN_WAV_DIR.mkdir(parents=True, exist_ok=True)

# 4. Temiz ses dosyalarının listelenmesi
def get_clean_audio_files(): 
    """Veri setindeki temiz .flac dosyalarını listeleyen fonksiyonu tanımlar."""
    flac_files = sorted(DATA_DIR.rglob("*.flac")) # veri setindeki tüm flac dosyalarını sıralı şekilde listeye alır

    if MAX_FILES is not None: # kullanılacak dosya sayısı sınırlandıysa bu bloğu çalıştır  
        flac_files = flac_files[:MAX_FILES]

    print(f"Bulunan ses dosyası sayısı: {len(flac_files)}")
    return flac_files

# get_clean_audio_files()

# 5. Tek bir ses dosyasını hazırlayan fonksiyon tanımla
def prepare_single_audio(input_file):
    """Tek bir .flac ses dosyasını okuyup .wav formatına çevirir"""

    relative_path = input_file.relative_to(DATASET_DIR) # ses dosyasının veri seti klasörüne göre göreli yolunu hesaplar
    output_name = "_".join(relative_path.with_suffix(".wav").parts) # yein wav adı oluşturur
    output_file = CLEAN_WAV_DIR / output_name # hazırlanan wav dosyasının kaydedileceği tam yolu oluşturur

    if output_file.exists(): # çıktı dosyası daha önce oluşturulduysas bu bloğu çalıştır
        return output_file # aynı dosyayı tekrar üretmeden mevcut dosya yolunu döndürür
    
    audio, _ = librosa.load(input_file, sr = SAMPLE_RATE, mono = True) # ses dosyasını 16 kHz ve mono olarak oku
    max_value = np.max(np.abs(audio)) # ses sinyalindeki en büyük mutlak genlik değerini hesapla

    if max_value > 0: # ses sinyali tamamen sessiz değilse
        audio = audio / max_value # ses sinyalini -1 ve 1 aralığına normalize et

    sf.write(output_file, audio, SAMPLE_RATE) # hazırlanan temiz sesi wav formatında kaydet
    return output_file

# 6. Tüm temiz ses dosyalarını .wav formatına dönüştür
def prepare_all_clean_audio(flac_files): 
    """tüm temiz ses dosyalarını wav formatına dönüştüren fonksiyon tanımlar"""
    created_files = [] # oluşturulan wav dosyalarının yollarını tuyacak boş liste

    for input_file in tqdm(flac_files): # tüm flac dosyalarını tara
        output_file = prepare_single_audio(input_file) # tek bir flac dosyasını wav formatına dönüştürür
        created_files.append(output_file)

    print(f"Hazırlanan wav dosyası sayısı: {len(created_files)}")
    print(f"Temiz wav klasörü: {CLEAN_WAV_DIR}")

# 7. Ana çalışma akışını oluştur 
def main():

    flac_files = get_clean_audio_files() # veri setindeki flac dosyalarını listeler

    if len(flac_files) == 0:
        return
    
    prepare_all_clean_audio(flac_files) # tüm ses dosyalarını wav formatına dönüştür

if __name__ == "__main__":
    main()