"""
Amaç:
    - Veri setindeki müzik dosyalarını okur ve Transformer modeline uygun Log-Mel Spectrogram formatına dönüştürür.
    - Öznitelikleri X_mel.npy ve sınıf etiketlerini y.npy olarak kaydet

Veri seti: metadata.csv

Adımlar:
    1. Gerekli kütüphanelerin içeriye aktarılması
    2. Proje klasör yollarının tanımlanması
    3. Ses işleme ayarlarının tanımlanması
    4. Metadata dosyasının okunması
    5. Ses dosyalarından Log-Mel Spectrogram özelliklerinin çıkarılması
    6. X ve y dizililerinin oluşturulması
    7. İşlenmiş verilerin kaydedilmesi
"""

# 1. Gerekli kütüphanelerin içeriye aktarılması
from pathlib import Path
import numpy as np
import pandas as pd
import librosa # ses dosyalarını okumak ve mel-spectrogram çıkarmak

# 2. Proje klasör yollarının tanımlanması
PROJECT_DIR = Path(__file__).resolve().parent # Bu python dosyasının bulunduğu klasörü proje ana klasörü olarak alır
PROCESSED_DIR = PROJECT_DIR / "data" / "processed" # işlenmiş dosya kaydedilecek klasör
METADATA_PATH = PROCESSED_DIR / "metadata.csv" # ses dosyası yolu ve etiket bilgilerinin kaydedileceği csv dosyası
X_MEL_PATH = PROCESSED_DIR / "X_mel.npy" # mel-spectrogram özelliklerinin kaydedileceği dosya yolu
Y_PATH = PROCESSED_DIR / "y.npy" # sınıf etiketlerinin kaydedileceği dosya yolu

PROCESSED_DIR.mkdir(parents=True, exist_ok=True) 

# 3. Ses işleme ayarlarının tanımlanması
SAMPLE_RATE = 22050 # ses dosyalarının okunacağı örnekleme frekansı
DURATION_SECONDS = 30 # her müzik dosyasının kaç saniyelik kısmının kullanılacağı
TARGET_LENGTH = SAMPLE_RATE * DURATION_SECONDS # her ses dosyasının sahip olması gereken toplam örnek sayısı
N_FFT = 2048 # kısa zamanlı fourier dönüşümü için pencere boyutunu tanımlar
HOP_LENGTH = 1024 # ardışık analiz pencereleri arasındaki örnek kayma miktarı
N_MELS = 128 # Mel-spectrogram üzerinde kullanılacak Mel frekans bandı sayısını tanımlar
EPSILON = 1e-8 # normalizasyon sırasında sıfıra bölünmeyi önlemek için küçük bir sabit

# 4. Metadata dosyasının okunması
metadata_df = pd.read_csv(METADATA_PATH)

# 5. Ses dosyalarından Log-Mel Spectrogram özelliklerinin çıkarılması
X_mel_list = [] # her ses dosyasından çıkarılan mel spectrogram verilerinin saklanması
y_list = [] # her ses dosyasından çıkarılan sınıf etiketlerinin saklanması

skipped_file_count = 0 # okunamayan veya işlenemeyen dosya sayısını takip etmek için
total_files = len(metadata_df) # metadata içinde toplam kaç ses dosyası olduğu

for index, row in metadata_df.iterrows(): # metadata tablosundaki her ses dosyası satırı üzerinde dolaşır
    file_path = Path(row["file_path"]) # ses dosyasının dosya yolunu Path nesnesine dönüştürür
    label = int(row["label"]) # ses dosyasının sayısal sınıf etiketi

    try: # ses dosyası okuma ve özellike çıkarma sırasında oluşabilecek hataları yakalamak için

        audio_signal, _ = librosa.load(file_path, sr = SAMPLE_RATE, mono=True)

        if len(audio_signal) < TARGET_LENGTH: # Ses sinyali hedef uzunluktan kısa ise kontrol edelim
            padding_amount = TARGET_LENGTH - len(audio_signal) # eksik olan örnek sayısını hesaplar
            audio_signal = np.pad(audio_signal, (0, padding_amount), mode = "constant") # kısa sesi sonuna sıfır ekleyerek hedef uzunluğa getir

        if len(audio_signal) > TARGET_LENGTH: # ses sinyali hedef uzunluktan büyükse
            audio_signal = audio_signal[:TARGET_LENGTH] # uzun sesi ilk 30 saniye olacak şekilde kırpar

        mel_spectrogram = librosa.feature.melspectrogram(
            y = audio_signal, # özellik çıkarılacak ses sinyali
            sr = SAMPLE_RATE, # frekans
            n_fft = N_FFT, # fourier dönüşümü pencere boyutu
            hop_length=HOP_LENGTH, # zaman ekseninde ki kayma miktarı
            n_mels = N_MELS # Mel frekans bandı sayısı
        )

        log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref = np.max) # mel spectrogram değerlerini desibel ölçeğine çeker
        log_mel_mean = np.mean(log_mel_spectrogram)
        log_mel_std = np.std(log_mel_spectrogram)
        log_mel_spectrogram = (log_mel_spectrogram - log_mel_mean) / (log_mel_std + EPSILON) # özellikleri ortalaması 0 ve standart sapması 1 olacak şekilde normalize edelim
        log_mel_spectrogram = log_mel_spectrogram.T # (X, zaman adımı) -> (zaman adımı, X)
        log_mel_spectrogram = log_mel_spectrogram.astype(np.float32)

        X_mel_list.append(log_mel_spectrogram)
        y_list.append(label)
    except Exception as e:
        print(e)
        
    if (index + 1) % 50 == 0:
        print(f"{index + 1}/{total_files} dosya işlendi.")

# 6. X ve y dizililerinin oluşturulması
X_mel = np.stack(X_mel_list, axis = 0)
y = np.array(y_list, dtype=np.int64)

# 7. İşlenmiş verilerin kaydedilmesi
np.save(X_MEL_PATH, X_mel)
np.save(Y_PATH, y)