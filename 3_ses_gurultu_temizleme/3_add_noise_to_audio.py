"""
Amaç:
    - Hazırlanan temiz .wav dosyalarına kontrollü bir şekilde yapay gürültü ekle
    - Temiz ses ve kirli ses çiftlerini oluşturarak autoencoder eğitim için veri hazırlama

Veri seti:
    - data/clean_wav: temiz ses verisi
    - bu dosyaya beyaz gürültü ekle ve kirli ses dosyaları oluştur

Adımlar:
    1. Gerekli kütüphanelerin içeriye aktarılması
    2. Proje klasör yollarının ve gürültü ayarlarının tanımlanması
    3. Gerekli klasörlerin oluşturulması
    4. Temiz wav dosyalarının listelenmesi
    5. SNR değerine göre gürültü ekleyen fonksiyonun tanımlanması
    6. Tek bir temiz ses dosyasından kirli ses oluşturulması
    7. Tüm temiz ses dosyalarına gürültü eklenmesi
    8. Ana çalışma akışının çalıştırılması
"""
# 1. Gerekli kütüphanelerin içeriye aktarılması
from pathlib import Path
import numpy as np
import librosa 
import soundfile as sf
from tqdm import tqdm

# 2. Proje klasör yollarının ve gürültü ayarlarının tanımlanması
BASE_DIR = Path(__file__).resolve().parent # Bu python dosyasının bulunduğu klasörü proje ana klasörü olarak alır
DATA_DIR = BASE_DIR / "data" # tüm veri dosyaları için ana klasör
CLEAN_WAV_DIR = DATA_DIR / "clean_wav" # hazırlanan temiz wav dosyalarının kaydedileceği klasör
NOISY_WAV_DIR = DATA_DIR / "noisy_wav" # gürültülü wav dosyalarının kaydedileceği klasör

SAMPLE_RATE = 16000 # seslerin 16 kHz örnekleme frekansından okunması
SNR_DB = 10 # gürültü seviyesini belirleyen sinyal gürültü oranının dB cinsinden tanımlanması
RANDOM_SEED = 42 # gürültü üretiminin tekrar edilebilir olması için rastgelelik değerlerini sabitler

# 3. Gerekli klasörlerin oluşturulması
NOISY_WAV_DIR.mkdir(parents=True, exist_ok=True)
np.random.seed(RANDOM_SEED) # rastgele gürültü üretimini sabit sonuç verecek hale getirelim

# 4. Temiz wav dosyalarının listelenmesi
def get_clean_wav_files():
    """temiz wav dosyalarını listeleyen fonksiyon"""
    clean_files = sorted(CLEAN_WAV_DIR.glob("*.wav")) # clean_wav klasöründe ki tüm wav dosyalarını sıralı şekilde listeye alır
    return clean_files

# 5. SNR değerine göre gürültü ekleyen fonksiyonun tanımlanması
def add_white_noise_with_snr(clean_audio, snr_db):
    """Temiz sese belirlenen SNR seviyesinde beyaz gürültü ekleyelim."""

    signal_power = np.mean(clean_audio**2) # temiz ses sinyalinden ortalama gücü hesaplar
    snr_linear = 10 ** (snr_db/10) # db cinsinden verilen SNR değerini doğrusal ölçeğe dönüştürür
    noise_power = signal_power / snr_linear # hedef snr değerine uygun gürültü gücü
    noise = np.random.normal(0, np.sqrt(noise_power), clean_audio.shape) # hesaplanan güce sahip beyaz gürültü
    noisy_audio = clean_audio + noise # temiz ses ile beyaz gürültüyü topla = kirli ses
    noisy_audio = np.clip(noisy_audio, -1, 1) # ses genliğini -1 ile 1 aralığında sınırlar

    return noisy_audio

# 6. Tek bir temiz ses dosyasından kirli ses oluşturulması
def crete_noisy_audio_file(clean_file): 
    """Tek bir temiz ses dosyasından kirli ses wav dosyası oluşturma"""

    output_file = NOISY_WAV_DIR / clean_file.name # kirli sesin temiz sesle aynı isimde kaydedileceği yol

    if output_file.exists(): 
        return output_file
    
    clean_audio, _ = librosa.load(clean_file, sr = SAMPLE_RATE, mono=True)
    noisy_audio = add_white_noise_with_snr(clean_audio, SNR_DB) # temiz sese belirtilen snr doğrultusunda gürültü ekler
    sf.write(output_file, noisy_audio, SAMPLE_RATE) # oluşturulan kirli sesi wav formatında kaydeder

    return output_file

# 7. Tüm temiz ses dosyalarına gürültü eklenmesi
def create_all_noisy_audio(clean_files): 
    """tüm temiz ses dosyalarından kirli ses dosyaları oluşturan fonksiyon tanımlar"""

    created_files = []

    for clean_file in tqdm(clean_files): # tüm temiz ses dosyaları üzerinde dolaş
        output_file = crete_noisy_audio_file(clean_file) # tek bir temiz sesten kirli ses dosyası oluştur
        created_files.append(output_file)

    print(f"Oluşturulan kirli wav dosyası sayısı: {len(created_files)}")
    print(f"Kirli wav klasörü: {NOISY_WAV_DIR}")

# 8. Ana çalışma akışının çalıştırılması
def main():
    clean_files = get_clean_wav_files() # temiz wav dosyalarını listele

    if len(clean_files) == 0: 
        return
    
    create_all_noisy_audio(clean_files) # tüm temiz ses dosyalarına gürültü ekler

if __name__ == "__main__":
    main()