"""
Amaç:
    ESC-50 veri setinden örnek bir ses dosyasını yükler, sesin dalga formunu ve mel spectogram görselini oluştur.

Adımlar:
    1. Gerekli kütüphanelerin içeriye aktarılması
    2. Veri seti klasör yollarının tanımlanması
    3. Metadata dosyasının okunması
    4. Örnek ses dosyasının seçilmesi
    5. Ses dosyasının yüklenmesi
    6. Ses bilgilerinin ekrana yazdırılması
    7. Waveform grafiğinin oluşturulması
    8. Mel Spectogram hesaplanması
    9. Mel Spectogram grafiğinin oluşturulması
    10. Ana çalışma akışının oluşturulması
"""

# 1. Gerekli kütüphanelerin içeriye aktarılması
import os 
import numpy as np 
import pandas as pd 
import librosa # ses dosyalarını yüklemek ve ses özelliklerini çıkarmak
import librosa.display # ses verilerini grafik üzerinde göstermek için kullanılıyor
import matplotlib.pyplot as plt 

# 2. Veri seti klasör yollarının tanımlanması
DATASET_DIR = "data/raw/ESC-50-master"
META_DATA = os.path.join(DATASET_DIR, "meta", "esc50.csv") # etiket dosyasının yolu
AUDIO_DIR = os.path.join(DATASET_DIR, "audio") # wav ses dosyalarının bulunduğu klasör
OUTPUT_DIR = "outputs" # grafiklerin kaydedileceği klasör

# 3. Metadata dosyasının okunması
def load_metadata():
    """ESC-50 metadata dosyasını okur ve dataframe olarak return eder."""
    return pd.read_csv(META_DATA)

# 4. Örnek ses dosyasının seçilmesi
def select_sample_audio(df, index = 0):
    """Veri setinden belirtilen index değerine göre örnek bir ses dosyası seç"""
    sample = df.iloc[index] 
    filename = sample["filename"]
    category = sample["category"]
    audio_path = os.path.join(AUDIO_DIR, filename) # ses dosyasının tam yolu
    return audio_path, filename, category

# 5. Ses dosyasının yüklenmesi
def load_audio(audio_path):
    """Ses dosyasını librosa ile yükler"""
    audio, sr = librosa.load(audio_path, sr = None) # ses dosyasını orijinal örnekleme oranıyla yükler
    return audio, sr # ses sinyalini, sr = sampling rate (örnekleme oranı)

# 6. Ses bilgilerinin ekrana yazdırılması
def show_audio_info(audio, sr, filename, category, audio_path):

    duration = len(audio)/sr # ses uzunluğunu saniye cinsinden hesaplar
    print(f"Örnek ses dosyası bilgisi: ")
    print(f"Dosya adı: {filename}")
    print(f"Sınıf: {category}")
    print(f"Dosya yolu: {audio_path}")
    print(f"Örnekleme oranı: {sr}")
    print(f"Ses uzunluğu: {round(duration,1)} saniye")
    print(f"Ses veri boyutu: {audio.shape}")

# 7. Waveform grafiğinin oluşturulması
def plot_waveform(audio, sr, category):
    """Ses sinyalinin waveform grafiğini oluştur ve kaydet"""

    os.makedirs(OUTPUT_DIR, exist_ok=True) # outputs klasörü yoksa oluştur

    plt.figure()
    librosa.display.waveshow(audio, sr = sr) # ses sinyalini zaman ekseninde görselleştir
    plt.title(f"Waveform - {category}")
    plt.xlabel("Zaman")
    plt.ylabel("Genlik")
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "sample_waveform.png")
    plt.savefig(save_path)
    plt.show()

# 8. Mel Spectogram hesaplanması
def create_mel_spectogram(audio, sr):
    """mel spectogram hesapla"""
    mel_spec = librosa.feature.melspectrogram(
        y = audio,
        sr = sr, # 44100 Hz sesin frekans çözünürlüğü ve modelin işleyebileceği maksimum frekans
        n_mels = 128, # mel bandı sayısı 
        n_fft = 2048, # hızlı fourier dönüşüm pencere boyutu 
        hop_length=512 # adım aralığı 
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref = np.max) # mel değerlerini desibel ölçeğine çevir
    return mel_spec_db

# 9. Mel Spectogram grafiğinin oluşturulması
def plot_mel_spectogram(mel_spec_db, sr, category):
    """Mel spectogram grafiğini oluşturur ve kaydeder"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.figure()
    librosa.display.specshow(
        mel_spec_db, sr = sr, hop_length=512, x_axis="time", y_axis="mel"
    )
    plt.colorbar(format = "%+2.0f dB") # renk skalasını desibel formatında ekler
    plt.title(f"Mel Spectogram - {category}")
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "sample_mel_spectogram.png")
    plt.savefig(save_path)
    plt.show() 

# 10. Ana çalışma akışının oluşturulması
def main():

    df = load_metadata() # metadata yükler
    audio_path, filename, category = select_sample_audio(df, index = 0) # ses dosyası seçer

    audio, sr = load_audio(audio_path) # ses dosyasını yükler

    show_audio_info(audio, sr, filename, category, audio_path)

    plot_waveform(audio, sr, category) # waveform grafiğinin oluşturulması

    mel_spec_db = create_mel_spectogram(audio, sr) # mel spectogram değerlerini hesaplar
    plot_mel_spectogram(mel_spec_db, sr, category)

if __name__ == "__main__":
    main()