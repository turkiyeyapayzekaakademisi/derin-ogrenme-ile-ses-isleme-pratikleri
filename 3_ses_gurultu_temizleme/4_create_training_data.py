"""
Amaç:
    - Temiz ve gürültülü ses dosyalarını eşleştirerek autoencoder modeli için eğitim verisi oluştur.
    - Gürültülü sesleri X, temiz sesleri y olarak ayarlanır

Veri seti:
    - data/clean_wav temiz konuşma sesleri
    - data/noisy_wav gürültü eklenmiş konuşma sesleri
    - Modelin girdisi gürültülü ses, çıktısı ise temiz ses olacaktır

Adımlar:
    1. Gereklü kütüphanelerin içeriye aktarılması
    2. Proje klasör yolları ve veri hazırlama ayarlarının tanımlanması
    3. Gerekli klasörlerin oluşturulması
    4. Temiz ve kirli ses dosyalarının eşleştirilmesi
    5. Tek bir ses dosyasını okuyan fonksiyon tanımla
    6. Ses sinyalini sabit uzunlukta parçalara ayıran fonksiyon tanımla
    7. Temiz ve gürültülü ses çiftlerinden veri oluştur
    8. Verinin train ve test olarak ayrılması
    9. Hazırlanan verilerin kaydedilmesi
    10. Ana çalışma akışının çalıştırılması
"""

# 1. Gereklü kütüphanelerin içeriye aktarılması
from pathlib import Path
import numpy as np
import librosa
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# 2. Proje klasör yolları ve veri hazırlama ayarlarının tanımlanması
BASE_DIR = Path(__file__).resolve().parent # Bu python dosyasının bulunduğu klasörü proje ana klasörü olarak alır
DATA_DIR = BASE_DIR / "data" # tüm veri dosyaları için ana klasör
CLEAN_WAV_DIR = DATA_DIR / "clean_wav" # hazırlanan temiz wav dosyalarının kaydedileceği klasör
NOISY_WAV_DIR = DATA_DIR / "noisy_wav" # gürültülü wav dosyalarının kaydedileceği klasör
PROCESSED_DATA_DIR = DATA_DIR / "processed" # model eğitime hazır numpy dosyalarının kaydedileceği klasör

SAMPLE_RATE = 16000 # 16 kHz örnekleme frekansı
DURATION = 2 # her eğitim örneğinin kaç saniyelik olacağını belirler
AUDIO_LENGTH = SAMPLE_RATE * DURATION # her eğitim örneğinin kaç örnekten oluşacağını hesaplar
TEST_SIZE = 0.2 # Verinin yüzde kaçının test veri seti olacağını belirler
RANDOM_STATE = 42 
MAX_AUDIO_FILES = 500 # Eğitim süresini makul tutmak için kullanılacak maksimum ses dosyası sayısı

# 3. Gerekli klasörlerin oluşturulması
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

# 4. Temiz ve kirli ses dosyalarının eşleştirilmesi
def get_audio_pairs(): 
    """Temiz ve gürültülü ses dosyalarını dosya adına göre eşleştiren fonksiyon tanımlar"""

    clean_files = sorted(CLEAN_WAV_DIR.glob("*.wav"))
    noisy_files = sorted(NOISY_WAV_DIR.glob("*.wav"))

    clean_file_dict = {file.name: file for file in clean_files} 
    noisy_file_dict = {file.name: file for file in noisy_files} 

    common_file_names = sorted(set(clean_file_dict.keys()) & set(noisy_file_dict.keys())) # hem temiz hem kirli klasörde bulunan ortak dosyaların adını bul

    if MAX_AUDIO_FILES is not None:
        common_file_names = common_file_names[:MAX_AUDIO_FILES]

    audio_pairs = [(clean_file_dict[name], noisy_file_dict[name]) for name in common_file_names] # temiz ve kirli dosya yollarını ikili çiftler halinde oluştur

    return audio_pairs

# 5. Tek bir ses dosyasını okuyan fonksiyon tanımla
def load_audio_file(file_path):
    """Tek bir wav dosyasını okuan fonksiyon"""

    audio, _ = librosa.load(file_path, sr = SAMPLE_RATE, mono = True) 
    audio = audio.astype(np.float32) 
    max_value = np.max(np.abs(audio)) # ses sinyalindeki en büyük mutlak genlik değerini hesaplar

    if max_value > 0:
        audio = audio / max_value # ses sinyalini -1 ile 1 arasına normalize eder

    return audio # okunan ve normalize edilen ses sinyali

# 6. Ses sinyalini sabit uzunlukta parçalara ayıran fonksiyon tanımla
def split_audio_into_chunks(audio):
    """Ses sinyalini sabit uzunluktaki parçalara ayırır"""

    chunks = [] # oluşturulacak ses parçalarını tutacak boş liste

    if len(audio) < AUDIO_LENGTH:
        padded_audio = np.pad(audio, (0, AUDIO_LENGTH - len(audio)), mode = "constant") # kısa sesi sıfırlar ile tamamlayarak hedef uzunluğa getirir
        chunks.append(padded_audio) # sesi parça listesine ekle
        return chunks
    
    chunk_count = len(audio) // AUDIO_LENGTH # ses sinyalinden kaç adet tam ğarça çıkarıldığını hesaplar

    for index in range(chunk_count): # her tam ses parçası için döngü başlatır
        start = index * AUDIO_LENGTH # parçanın başlangıç indeksi
        end = start + AUDIO_LENGTH # parçanın bitiş indeksi
        chunk = audio[start:end]
        chunks.append(chunk)

    return chunks # sabit uzunluktaki ses parçaları

# 7. Temiz ve gürültülü ses çiftlerinden veri oluştur
def create_dataset(audio_pairs):
    """Temiz ve kirli ses çiftlerinden X ve y veri setlerini oluştur"""

    X = [] # kirli ses parçalarını tutacak boş liste
    y = [] # temiz ses parçalarını tutacak boş liste

    for clean_file, noisy_file in tqdm(audio_pairs):

        clean_audio = load_audio_file(clean_file) # temiz ses dosyasını okur
        noisy_audio = load_audio_file(noisy_file) # kirli ses dosyasını okur

        min_length = min(len(clean_audio), len(noisy_audio)) # en kısa olanı bul
        clean_audio = clean_audio[:min_length] # temiz sesi ortak uzunluğa getir
        noisy_audio = noisy_audio[:min_length] # kirli sesi ortak uzunluğa getir

        clean_chunks = split_audio_into_chunks(clean_audio) 
        noisy_chunks = split_audio_into_chunks(noisy_audio)

        for clean_chunk, noisy_chunk in zip(clean_chunks, noisy_chunks):
            X.append(noisy_chunk) # kirli ses parçaları X e 
            y.append(clean_chunk) # temiz ses parçaları y ye

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    X = np.expand_dims(X, axis = -1) # kirli ses verisine Conv1D için kanal boyutu ekler
    y = np.expand_dims(y, axis = -1)

    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")

    return X, y

# 8. Verinin train ve test olarak ayrılması
def split_dataset(X, y): 
    """X ve y verilerini train ve test olarak ayıran fonksiyon tanımlayalım"""

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=True)

    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")

    return X_train, X_test, y_train, y_test

# 9. Hazırlanan verilerin kaydedilmesi
def save_dataset(X_train, X_test, y_train, y_test):
    """Hazırlanan train ve test verilerini kaydeder"""
    np.save(PROCESSED_DATA_DIR / "X_train.npy", X_train)
    np.save(PROCESSED_DATA_DIR / "X_test.npy", X_test)
    np.save(PROCESSED_DATA_DIR / "y_train.npy", y_train)
    np.save(PROCESSED_DATA_DIR / "y_test.npy", y_test)

# 10. Ana çalışma akışının çalıştırılması
def main():
    """Ana çalışma akışını belirle"""

    audio_pairs = get_audio_pairs() # temiz ve kirli ses dosyalarını eşleştir

    if len(audio_pairs) == 0:
        print(f"audio_pairs len: {len(audio_pairs)}")
        return # programı sonlandır
    
    X, y = create_dataset(audio_pairs) # temiz ve kirli ses çiftelerinden X ve y oluştur
    X_train, X_test, y_train, y_test = split_dataset(X, y) # train test split
    save_dataset(X_train, X_test, y_train, y_test) # kaydet

if __name__ == "__main__":
    main()