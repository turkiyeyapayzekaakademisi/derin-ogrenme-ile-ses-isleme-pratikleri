"""
Amaç:
    metadata.csv içindeki ses dosyalarını okuyarak LSTM modeli için MFCC zaman dizisi özellikleri çıkaralım.
    Elde edilen MFCC özellikleri, sonraki aşamada insan sesinden duygu sınıflandırması yapan LSTM modeli inputu olur.

Veri seti: metadata.csv

Adımlar:
    1. Gerekli kütüphanelerin içeriye aktarılması
    2. Proje klasör yolları ve MFCC çıkarım ayarları tanımlama
    3. Metadata dosyası oku
    4. Ses dosyalarını yükle
    5. LSTM modeli için MFCC zaman dizisi özellikleri çıkaralım
    6. Farklı uzunluktaki MFCC özelliklerini aynı boyuta getir
    7. Tüm ses dosyaları için MFCC çıkarım süreci çalıştırma
    8. Çıkarılan MFCC özelliklerini numpy formatında kaydet
    9. Örnek MFCC görseli oluştur
"""

# 1. Gerekli kütüphanelerin içeriye aktarılması
from pathlib import Path # Dosya ve klasör yollarını yönetmek için
import librosa # ses dosyalarını okumak ve ses sinyali özelliklerini analiz etmek
import librosa.display # waveform görselleştirme
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 2. Proje klasör yolları ve MFCC çıkarım ayarları tanımlama
PROJECT_DIR = Path(__file__).resolve().parent # çalışan python dosyanın bulunduğu klasörü proje ana dizini olarak alır
DATA_DIR  = PROJECT_DIR / "data" # ham ve işlenmiş veriler için ana veri klasörü
PROCESSED_DATA_DIR = DATA_DIR / "processed" # metadata gibi işlenmiş dosyaların kaydedileceği klasör 
PLOTS_DIR = PROJECT_DIR / "outputs" / "plots" # grafik çıktıları klasörü

METADATA_PATH = PROCESSED_DATA_DIR / "metadata.csv"
MFCC_FEATURES_PATH = PROCESSED_DATA_DIR / "mfcc_features.npy" # lstm için gerekli olan mfcc özelliklerinin kayıt yolu
LABELS_PATH = PROCESSED_DATA_DIR / "labels.npy"

PLOTS_DIR.mkdir(parents=True, exist_ok=True) # Grafik çıktı klasörü yoksa oluştur
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True) # processed klasörü yoksa oluştur

SAMPLE_RATE = 22050 # tüm sesleri aynı örnekleme frekansına getirerek standart bir veri yapısı oluşturuyoruz
DURATION = 3.0 # her ses kaydını 3 saniyelik standart uzunlukta ele alalım
TARGET_LENGTH = int(SAMPLE_RATE * DURATION) # 3 saniyelik sesin toplam örnek sayısı

N_MFCC = 40 # mfcc çıktısında kullanılacak katsayı
N_FFT = 2048 # grekans dönüşümü sırasında kullanılacak pencere boyutu
HOP_LENGTH = 512 # ardışık analiz pencereleri arasındaki adım uzunluğu
MAX_TIME_STEPS = 130 # MFCC çıktılarında kullnılacak sabit zaman adımı sayısı

# 3. Metadata dosyası oku
def load_metadata(metadata_path: Path) -> pd.DataFrame:
    """Metadata.csv okur"""

    return pd.read_csv(metadata_path)

# 4. Ses dosyalarını yükle
def load_audio(file_path: str) -> np.ndarray:
    """Ses dosyasını yükle"""

    signal, _ = librosa.load(file_path, sr = SAMPLE_RATE) # ses dosyasını belirlenen örnekleme frekansı ile yükle

    if len(signal) < TARGET_LENGTH: # Ses kaydı hedef uzunluktan kısaysa sıfırlarla tamamla
        padding_length = TARGET_LENGTH - len(signal) # eksik kalan örnek sayısını hesapla
        signal = np.pad(signal, (0, padding_length), mode="constant") # kısa sesleri sıfır ile doldur
    else:
        signal = signal[:TARGET_LENGTH] # Uzun sesleri heded uzunluğuna kırparak standartlaştırıyoruz

    return signal

# 5. LSTM modeli için MFCC zaman dizisi özellikleri çıkaralım
def extract_mfcc(signal: np.ndarray) -> np.ndarray:
    """ses sinyalinden MFCC zaman dizisi özelliğini çıkarmak"""

    mfcc = librosa.feature.mfcc(
        y = signal,
        sr = SAMPLE_RATE,
        n_mfcc = N_MFCC,
        n_fft = N_FFT,
        hop_length = HOP_LENGTH
    ) # Ses sinyalinden konuşma ve ton karakteristiğini temsil eden mfcc katsayılarını çıkart

    mfcc = fix_feature_length(feature=mfcc, max_time_steps= MAX_TIME_STEPS) # mfcc zaman boyutunu tüm örneklerde aynı uzunluğa getir

    mfcc = mfcc.T # LSTM giriş formatına uygun olması için veriyi (time_steps, features) şekline çeviriyoruz

    return mfcc

# 6. Farklı uzunluktaki MFCC özelliklerini aynı boyuta getir
def fix_feature_length(feature: np.ndarray, max_time_steps: int) -> np.ndarray:
    """Özellik matrisinin zaman boyutunu sabit uzunluğa getir"""

    current_time_steps = feature.shape[1] # özellik matrisinin mevcut zaman adımı sayısını alalım

    if current_time_steps < max_time_steps: # zaman boyutu hedef değerden kısaysa padding uygula
        padding_width = max_time_steps - current_time_steps # eklenecek zaman adımı sayısı
        feature = np.pad(feature, pad_width=((0,0), (0, padding_width)), mode = "constant")
    else:
        feature = feature[:, : max_time_steps] # uzun özellikleri hedef zaman adımı sayısına kırpalım

    return feature

# 7. Tüm ses dosyaları için MFCC çıkarım süreci çalıştırma
def extract_features_from_dataset(metadata_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Metadata tablosundaki tüm ses dosyalarından MFCC ve etiket dizilerini üretir."""

    mfcc_features = [] # LSTM modelin kullanılacak MFCC zaman dizilerini bu listede tutarız 
    labels = [] # her ses dosyasına karşılık geken duygu etiketlerini bu listede tutarız

    total_files = len(metadata_df) # işlenecek toplam ses dosyası sayısı

    for index, row in metadata_df.iterrows(): # metadata tablosundaki her ses dosyasını sırayla işler

        file_path = row["file_path"]
        label = row["emotion"]

        try: 
            signal = load_audio(file_path) # ses dosyasını yükle
            mfcc = extract_mfcc(signal) # lstm için mfcc türetme

            mfcc_features.append(mfcc)
            labels.append(label)

        except Exception as e:
            print(e)

        if (index + 1) % 100 == 0:
            print(f"işlenen dosya sayısı: {index+1}/{total_files}")

    mfcc_features_array = np.array(mfcc_features, dtype=np.float32) 
    labels_array = np.array(labels) 

    return mfcc_features_array, labels_array

# 8. Çıkarılan MFCC özelliklerini numpy formatında kaydet
def save_features(mfcc_features: np.ndarray, labels: np.ndarray) -> None:
    """Çıkarılan mfcc özelliklerini ve etiketleri numpy dosyası olarak kaydeder"""

    np.save(MFCC_FEATURES_PATH, mfcc_features)
    np.save(LABELS_PATH, labels)

# 9. Örnek MFCC görseli oluştur
def plot_sample_mfcc(mfcc_features: np.ndarray, labels: np.ndarray) -> None:
    """Örnek mfcc görseli oluştur"""

    sample_index = 0
    sample_label = labels[sample_index] # seçilen örneğin duygu etiketi

    plt.figure()
    # lstm için hazırlanan mfcc zaman dizisini görselleştirmek için features x i time formatında gösteriyoruz
    librosa.display.specshow(mfcc_features[sample_index].T, sr=SAMPLE_RATE, hop_length=HOP_LENGTH, x_axis="time")
    plt.colorbar()
    plt.title(f"Örnek MFCC - Duygu {sample_label}")
    plt.xlabel("Zaman")
    plt.ylabel("MFCC katsayıları")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "sample_mfcc.png")
    plt.close()


def main() -> None:
    
    metadata_df = load_metadata(METADATA_PATH) # metatada dosyası okur

    mfcc_features, labels = extract_features_from_dataset(metadata_df) # tüm ses dosylarından LSTM için mfcc çıkart

    save_features(mfcc_features, labels) # sonradan kullanmak için kaydet

    plot_sample_mfcc(mfcc_features, labels) # görsel kontrol

if __name__ == "__main__":
    main()