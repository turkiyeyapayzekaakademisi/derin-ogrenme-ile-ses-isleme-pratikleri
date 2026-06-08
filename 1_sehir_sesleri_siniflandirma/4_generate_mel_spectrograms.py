"""
Amaç:
    ESC-50 ses dosyalarını augmentation destekli Mel Spectogram görsellerine dönüştür.

Adımlar:
    1. Gerekli kütüphanerin içeriye aktarılması
    2. Veri seti ve çıktı klasör yollarını tanımla
    3. Fold değerine göre veri bölümünün belirlenmesi (train-val-test)
    4. Ses verisine noise augmentation uygula
    5. Ses verisine time shift augmentation uygulama 
    6. Ses verisine pitch shift uygulama
    7. Mel spectogram görselinin kaydedilmesi
    8. Bir ses dosyasından orijinal ve arttırılmış görsellerin üretilmesi
    9. Metadata dosyasının okunması
    10. Tüm ses dosyalarının işlenmesi
    11. Ana çalışma akışının oluşturulması
"""

# 1. Gerekli kütüphanerin içeriye aktarılması
import os
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
from tqdm import tqdm

# 2. Veri seti ve çıktı klasör yollarını tanımla
DATASET_DIR = "data/raw/ESC-50-master"
META_DATA = os.path.join(DATASET_DIR, "meta", "esc50.csv") # etiket dosyasının yolu
AUDIO_DIR = os.path.join(DATASET_DIR, "audio") # wav ses dosyalarının bulunduğu klasör
OUTPUT_DIR = "data/processed/mel_spectrograms" # mel spectrogram görüntülerinin kaydedileceği klasör

# 3. Fold değerine göre veri bölümünün belirlenmesi (train-val-test)
def get_split_name(fold):
    """Fold değerine göre train, val veya test bilgisini döndürür."""
    if fold in [1,2,3]: # train eğitim veri seti olarak kullan
        return "train"
    elif fold == 4: # validasyon doğrulama veri seti
        return "val"
    elif fold == 5: # test veri seti
        return "test"
    else: raise ValueError("Beklenmeyen bir fold değeri oluştu.")

# 4. Ses verisine noise augmentation uygula
def add_noise(audio, noise_factor = 0.005):
    """Ses sinyaline düşük seviyeli rastgele gürültü ekle"""
    noise = np.random.randn(len(audio)) # ses uzunluğu kadar rastgele gürültü üretir
    augmented_audio = audio + noise_factor * noise # orijinal sese düşük seviyeli gürültü ekler
    return augmented_audio.astype(np.float32) # ses verisini float32 formatında döndürür

# 5. Ses verisine time shift augmentation uygulama 
def time_shift(audio, shift_max = 0.2):
    """Ses sinyalini zaman eksenine göre sağa yada sola kaydırır"""
    shift = int(np.random.uniform(-shift_max, shift_max)*len(audio)) # rastgele kaydırma miktarı hesaplar
    augmented_audio = np.roll(audio, shift) # ses sinyalini hesaplanan miktar kadar kaydırır
    return augmented_audio.astype(np.float32) 

# # 6. Ses verisine pitch shift augmentation uygulanması
# def pitch_shift(audio, sr, n_steps=2):
#     """Sesin perde değerini hafifçe değiştirir."""
#     step = np.random.uniform(-n_steps, n_steps)  # Rastgele pitch değişim miktarı belirler.
#     augmented_audio = librosa.effects.pitch_shift(y=audio, sr=sr, n_steps=step)  # Sesin pitch değerini değiştirir.
#     return augmented_audio.astype(np.float32)  # Ses verisini float32 formatında döndürür.

# 6. Ses verisine pitch shift uygulama
def pitch_shift(audio, sr, n_steps=2):
    """
    Sesin perde değerini basit yöntemle değiştirir.
    
    Not:
        Burada librosa.effects.pitch_shift kullanmıyoruz.
        Çünkü bazı Windows ortamlarında librosa.effects sklearn import ediyor
        ve DLL engelleme hatası oluşabiliyor.
    """

    # -n_steps ile +n_steps arasında rastgele perde değişimi seçilir
    step = np.random.uniform(-n_steps, n_steps)

    # Yarım ses/semitone dönüşüm oranı hesaplanır
    rate = 2 ** (step / 12)

    # Orijinal indeksler
    original_indices = np.arange(len(audio))

    # Yeni örnekleme indeksleri
    shifted_indices = np.arange(0, len(audio), rate)

    # Ses sinyali yeni indekslere göre interpolate edilir
    shifted_audio = np.interp(
        shifted_indices,
        original_indices,
        audio
    )

    # Eğer yeni ses kısa kaldıysa sona sıfır ekle
    if len(shifted_audio) < len(audio):
        pad_length = len(audio) - len(shifted_audio)
        shifted_audio = np.pad(shifted_audio, (0, pad_length))

    # Eğer yeni ses uzun olduysa kırp
    else:
        shifted_audio = shifted_audio[:len(audio)]

    return shifted_audio.astype(np.float32)

# 7. Mel spectogram görselinin kaydedilmesi
def save_mel_spectrogram_image(audio, sr, save_path):
    """Ses sinyalini mel spectogram görseline dönüştürür ve kaydeder"""
    mel_spec = librosa.feature.melspectrogram(
        y = audio,
        sr = sr, # 44100 Hz sesin frekans çözünürlüğü ve modelin işleyebileceği maksimum frekans
        n_mels = 128, # mel bandı sayısı 
        n_fft = 2048, # hızlı fourier dönüşüm pencere boyutu 
        hop_length=512 # adım aralığı 
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref = np.max) # mel değerlerini desibel ölçeğine çevir

    plt.figure()
    plt.axis("off")
    librosa.display.specshow(mel_spec_db, sr = sr, hop_length=512, cmap = "magma")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close() # bellek kullanımını azaltmak için grafiği kapatır 

# 8. Bir ses dosyasından orijinal ve arttırılmış görsellerin üretilmesi
def create_images_for_audio(audio_path, output_dir, base_filename, split_name):
    """Bir ses dosyasından orijinal ve augmented Mel Spectogram görselleri üretir"""

    audio, sr = librosa.load(audio_path, sr = 22050) # ses dosyasını 22050 örnekleme oranıyla yükle
    orijinal_path = os.path.join(output_dir, base_filename + "_orijinal.png") # orijinal görselin kayıt yolu

    if not os.path.exists(orijinal_path): # orijinal görsel daha önce üretilmemişse kontrolu
        save_mel_spectrogram_image(audio, sr, orijinal_path) # orijinal sesten mel spectogram görseli üretir

    if split_name != "train": # train dışındaki verilerde augmentation yapılmaz
        return # validasyon ve test için sadece orijinal görsel üretilir
    
    noise_path = os.path.join(output_dir, base_filename + "_noise.png") # noise augmentation görsel yolu
    shift_path = os.path.join(output_dir, base_filename + "_shift.png") # shift augmentation görsel yolu
    pitch_path = os.path.join(output_dir, base_filename + "_pitch.png") # pitch augmentation görsel yolu
    
    if not os.path.exists(noise_path): # noise görseli daha önce üretilmemişse kontrol eder
        noise_audio = add_noise(audio) # orijinal sese gürültü ekler
        save_mel_spectrogram_image(noise_audio, sr, noise_path) # gürültülü sesten mel spectogram görseli üret

    if not os.path.exists(shift_path): # time shift görseli daha önce üretilmemişse kontrol eder
        shifted_audio = time_shift(audio) # orijinal sesi zaman ekseninde kaydırır
        save_mel_spectrogram_image(shifted_audio, sr, shift_path) # kaydırılmış sesten mel spectogram görseli üret

    if not os.path.exists(pitch_path): # pitch görseli daha önce üretilmemişse kontrol eder
        pitched_audio = pitch_shift(audio, sr) # orijinal sesi kalınlaştır yada incelt
        save_mel_spectrogram_image(pitched_audio, sr, pitch_path) # pitch değiştirilmiş sesten mel spectogram görseli üret

# 9. Metadata dosyasının okunması
def load_metadata():
    return pd.read_csv(META_DATA)

# 10. Tüm ses dosyalarının işlenmesi
def process_audio_files(df):
    """tüm ses dosyalarını sırayla mel spectogram görsellerine dönüştürür"""

    os.makedirs(OUTPUT_DIR, exist_ok=True) 

    for _, row in tqdm(df.iterrows(), total = len(df)):
        filename = row["filename"]
        category = row["category"]
        fold = row["fold"]
        split_name = get_split_name(fold)
        audio_path = os.path.join(AUDIO_DIR, filename)
        class_dir = os.path.join(OUTPUT_DIR, split_name, category)
        base_filename = filename.replace(".wav", "") # uzantısız dosya adını oluştur

        os.makedirs(class_dir, exist_ok=True) # sınıf klasörü yoksa oluştur

        if not os.path.exists(audio_path): # ses dosayı var mı yok mu kontrol
            print("ses dosyası bulunamadı")
            continue

        create_images_for_audio(audio_path, class_dir, base_filename, split_name) # ses dosyasından görsel üret

# 11. Ana çalışma akışının oluşturulması
def main():
    """Mel spectogram üretme adımlarını sırasıyla çalıştır"""
    df = load_metadata()
    process_audio_files(df) # tüm ses dosyalarını mel spectogram görsellerine dönüştür
    print(f"Kayır klasörü: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()