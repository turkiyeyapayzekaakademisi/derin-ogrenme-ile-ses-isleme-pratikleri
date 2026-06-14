"""
Amaç:
    - MFCC özelliklerini ve duygu etiketlerini LSTM model eğitimine uygun train/test formatına çevirelim
    - Sadece eğitim veri setine data augmentation uygulayarak LSTM modelinin daha genellenebilir örüntüler öğrenmesini sağlayalım

Veri seti: .npy uzantılı veriyi kullanalım 

Adımlar:
    1. Gerekli kütüphanelerin içeriye aktarılması
    2. Proje klasör yollarının tanımlanması ve veri hazırlama ayarları
    3. MFCC özellik ve etiket dosyalarını yükle
    4. Duygu etiketlerini sayısal sınıf değerlerine dönüştür
    5. MFCC verilerini normalize edelim
    6. LSTM verisini train - test olarak ayır
    7. Train verisine LSTM için data augmentation uygulama
    8. Hazırlanan veri setlerini numpy dosyaları olarak kaydet
    9. Label encoder nesnesini kaydet
"""

# 1. Gerekli kütüphanelerin içeriye aktarılması
from pathlib import Path 

import joblib # label encoder nesnesi kaydetmek için
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import LabelEncoder # duygu etiketlerini sayısal sınıf değerlerine çevirmek için

# 2. Proje klasör yollarının tanımlanması ve veri hazırlama ayarları
PROJECT_DIR = Path(__file__).resolve().parent # çalışan python dosyanın bulunduğu klasörü proje ana dizini olarak alır
DATA_DIR  = PROJECT_DIR / "data" # ham ve işlenmiş veriler için ana veri klasörü
PROCESSED_DATA_DIR = DATA_DIR / "processed" # metadata gibi işlenmiş dosyaların kaydedileceği klasör 
MODELS_DIR = PROJECT_DIR / "models" # model dosyaları ve label encoder gibi nesnelerin kaydedileceği klasör

MFCC_FEATURES_PATH = PROCESSED_DATA_DIR / "mfcc_features.npy" # lstm için gerekli olan mfcc özelliklerinin kayıt yolu
LABELS_PATH = PROCESSED_DATA_DIR / "labels.npy"

LSTM_X_TRAIN_PATH = PROCESSED_DATA_DIR / "lstm_X_train.npy" # lstm eğitim giriş verisi kaydedilecek yol
LSTM_X_TEST_PATH = PROCESSED_DATA_DIR / "lstm_X_test.npy" # lstm test giriş verisi kaydedilecek yol
Y_TRAIN_PATH = PROCESSED_DATA_DIR / "y_train.npy" # eğitim etiketlerinin kaydedileceği yol
Y_TEST_PATH = PROCESSED_DATA_DIR / "y_test.npy" # test etiketlerinin kaydedileceği yol

LABEL_ENCODER_PATH = MODELS_DIR / "label_encoder.pkl" # duygu sınıf isimlerini saklayacak label encoder dosya yolu

TEST_SIZE = 0.2 #  veri setinin %20 lik kısmını test seti olarak ayıralım
RANDOM_STATE = 42 # train test ayrımının her çalıştırmada aynı sonucu üretmesi

AUGMENTATION_FACTOR = 3 # her orijinal örnek için 3 adet arttırılmış örnek üret
LSTM_NOISE_STD = 0.02 # MFCC zaman dizisine eklenecek rastgele gürültünün şiddeti
MAX_LSTM_TIME_SHIFT = 10 # MFCC için zaman ekseninde maksimum kaydırma miktarı
MAX_LSTM_FEATURE_SHIFT = 3 # MFCC katsayısı ekseninde maksimum kaydırma miktarı

PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True) 

np.random.seed(RANDOM_STATE) # data augmentation işlemlerinin daha tekrarlanabilir olması için numpy rastgeleliğini sabitle

# 3. MFCC özellik ve etiket dosyalarını yükle
def load_feature_files() -> tuple[np.ndarray, np.ndarray]:
    """MDCC ve label dosyalarını yükler"""

    required_files = [
        MFCC_FEATURES_PATH, 
        LABELS_PATH
    ] # bu dosyanın çalışması için gerekli olan .npy dosyaları

    for file_path in required_files: # gerekli dosyaların kontrol edilmesi
        if not file_path.exists():  # dosyalardan biri eksikse sonraki adımlar doğru çalışmayacağı için hata verelim
            raise FileNotFoundError(f"Gerekli dosya bulunamadı: {file_path}")
        
    mfcc_features = np.load(MFCC_FEATURES_PATH)
    labels = np.load(LABELS_PATH, allow_pickle=True) # metin duygu etiketlerini numpy dizisi olarak yükle

    return mfcc_features, labels

# 4. Duygu etiketlerini sayısal sınıf değerlerine dönüştür
def encode_labels(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Metin duygu etiketlerini sayısal sınıf değerlerine dönüştür"""

    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels) # metin etiketleri 0'dan başlayan sayısal sınıf değerlerine dönüştürülüyor

    return encoded_labels, label_encoder # sayısal etiketleri ve eğitilmiş LabelEncoder nesnesini döndür

# 5. MFCC verilerini normalize edelim
def normalize_features(features: np.ndarray) -> np.ndarray:
    """Özellik dizisini ortalama ve standart sapmaya göre normalize eder"""

    mean_value = np.mean(features)
    std_value = np.std(features)

    normalized_feautres = (features - mean_value) / (std_value + 1e-8)  # sınıf ortalama ve birim varyansa yakın hale getir
    return normalized_feautres.astype(np.float32) # model eğitimi bellek verimliliği için veriyi float32 formatına dönüştür

# 6. LSTM verisini train - test olarak ayır
def split_training_data(lstm_features: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    """LSTM verisinin sınıf dağılımı koruyarak train ve test olarak ayır"""

    X_train, X_test, y_train, y_test = train_test_split(lstm_features, labels, test_size=TEST_SIZE, random_state=RANDOM_STATE,stratify=labels)

    return X_train, X_test, y_train, y_test 

# 7. Train verisine LSTM için data augmentation uygulama
def add_lstm_noise(mfcc_sequence: np.ndarray) -> np.ndarray:
    """MFCC zaman dizisine küçük rastgele gürültü ekler"""

    noise = np.random.normal(loc = 0.0, scale = LSTM_NOISE_STD, size = mfcc_sequence.shape) # rastgele gürültü

    augmented_mfcc = mfcc_sequence + noise # LSTM modelini küçük mFCC değişimlerine saha dayanıklı hale getirmek için gürültü ekle

    return augmented_mfcc.astype(np.float32)

def apply_lstm_time_shift(mfcc_sequence: np.ndarray) -> np.ndarray:
    """MFCC zaman dizisini zaman ekseninde kaydır"""

    shift_value = np.random.randint(-MAX_LSTM_TIME_SHIFT, MAX_LSTM_TIME_SHIFT + 1) # zaman ekseninde uygulanacak rastgele kaydırma miktarı

    augmented_mfcc = np.roll(mfcc_sequence, shift=shift_value, axis = 0) # MFCC dizisini zaman ekseninde kaydırarak konuşma başlangıç konumuna bağımlılığı azaltmak

    return augmented_mfcc.astype(np.float32) 

def apply_lstm_feature_shift(mfcc_sequence: np.ndarray) -> np.ndarray:
    """MFCC katsayı ekseninde küçük kaydırma uygulayarak ton değişimini simüle eder"""

    shift_value = np.random.randint(-MAX_LSTM_FEATURE_SHIFT, MAX_LSTM_FEATURE_SHIFT + 1) # MFCC katsayı ekseninde uygulanacak rastgele kaydırma miktarı

    augmented_mfcc = np.roll(mfcc_sequence, shift_value, axis = 1) # MFCC katsayılarını kaydırarak ton da küçük değişiklikler yapalım

    return augmented_mfcc.astype(np.float32)

def augment_single_lstm_sample(mfcc_sequence: np.ndarray) -> np.ndarray:
    """Tek bir diziye rastgele data augmentation uygular"""

    augmented_mfcc = mfcc_sequence.copy()

    if np.random.rand() < 0.7: # Eğitim örneklerinin %70 ine küçük gürültü ekle
        augmented_mfcc = add_lstm_noise(augmented_mfcc )

    if np.random.rand() < 0.7: # Eğitim örneklerinin %70 ine zaman ekseninde kaydırma
        augmented_mfcc = apply_lstm_time_shift(augmented_mfcc)

    if np.random.rand() < 0.5: # Eğitim örneklerinin %50 ine MFCC katsayı ekseninde küçük kaydırma
        augmented_mfcc = apply_lstm_feature_shift(augmented_mfcc)

    return augmented_mfcc.astype(np.float32)   

def augment_training_data(X_train: np.ndarray, y_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """LSTM train verisini aynı etiket düzeyiyle arttırır"""

    augmented_features = [X_train] # orijinal LSTM eğitim verisini de eğitim setinde tutalım
    augmented_labels = [y_train] # orijinal eğitim etiketleri de eğitim setinde tutalım

    for _ in range(AUGMENTATION_FACTOR): # her orijinal örnek için belirlenen sayıda artırılmış örnek üret

        augmented_batch = [] # augmented için geçici liste

        for mfcc_sequence in X_train: # eğitim setindeki her MFCC zaman dizisini tek tek işle
            augmented_sample = augment_single_lstm_sample(mfcc_sequence)
            augmented_batch.append(augmented_sample) # artırılmış örneği geçici listeye ekle

        augmented_features.append(np.array(augmented_batch, dtype=np.float32))  # bu turdaki LSTM örneklerini ana listeye ekler
        augmented_labels.append(y_train.copy()) # etiketleri kopyala

    X_train_augmented = np.concatenate(augmented_features, axis = 0) # orijinal ve artırılmış LSTM training verisini birleştir
    y_train_augmented = np.concatenate(augmented_labels, axis=0) # orijinal ve artırılmış eğitim etiketlerini birleştir

    shuffle_indices = np.random.permutation((len(y_train_augmented))) # eğitim sırasında sıralı örnek etkisini azaltmak için
    X_train_augmented = X_train_augmented[shuffle_indices] # eğitim verisini rastgele sıraya sokuyoruz
    y_train_augmented = y_train_augmented[shuffle_indices]

    return X_train_augmented, y_train_augmented

# 8. Hazırlanan veri setlerini numpy dosyaları olarak kaydet
def save_prepared_data(X_train: np.ndarray, X_test:np.ndarray, y_train: np.ndarray, y_test: np.ndarray) -> None:
    """Hazırlanan eğitim ve test verilerini numpy formatında kaydet"""
    np.save(LSTM_X_TRAIN_PATH, X_train)
    np.save(LSTM_X_TEST_PATH, X_test)
    np.save(Y_TRAIN_PATH, y_train)
    np.save(Y_TEST_PATH, y_test)

# 9. Label encoder nesnesini kaydet
def save_label_encoder(label_encoder: LabelEncoder) -> None:
    """LabelEncoder nesnesini dosyaya kaydeder"""

    joblib.dump(label_encoder, LABEL_ENCODER_PATH)


def main() -> None:
    """LSTM model eğitimine uygun veri hazırlama sürecini başlatır"""
    print("LSTM veri hazırlama süreci başlatıldı")

    mfcc_features, labels = load_feature_files() # MFCC özellikleri ve duygu etiketi dosyasını yükler
    print("Yüklenen veri boyutları")
    print(f"MFCC shape: {mfcc_features.shape}")
    print(f"Labels shape: {labels.shape}")

    encoded_labels, label_encoder = encode_labels(labels) # metin duygu etiketlerini sayısal sınıf değerlerine dönüştürür
    print("Sınıf bilgileri")
    print(f"Sınıf isimleri: {list(label_encoder.classes_)}") # sayısal etiketlerin hangi duygu sınıflarına karşılık geldiğini gösterir
    print(f"Sınıf sayısı: {len(label_encoder.classes_)}") # modelin kaç farklı duygu sınıfını tahmin etmeye çalıştığı

    lstm_features = normalize_features(mfcc_features) # normalizasyon
    print("Model giriş formatı")
    print(f"LSTM giriş shape: {lstm_features.shape}") # LSTM modelinin alacağı verinin yapısı

    X_train, X_test, y_train, y_test = split_training_data(lstm_features, encoded_labels)
    print("Augmentation öncesi train-test veri boyutları")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")

    X_train, y_train = augment_training_data(X_train, y_train) # data augmentation
    print("Augmentation sonrası train-test veri boyutları")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")

    save_prepared_data(X_train, X_test, y_train, y_test) # hazırlanan tüm eğitim ve test verilerini kaydet

    save_label_encoder(label_encoder)

if __name__ == "__main__":
    main()

