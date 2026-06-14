"""
Amaç:
    - MFCC zaman dizilerini kullanarak LSTM tabanlı bir duygu sınıflandırma modeli 

Veri seti: x_train, x_test, y_tran, y_test

Adımlar:
    1. Gerekli kütüphanelerin içeriye aktarılması
    2. Proje klasör yollarını ve eğitim ayarlarını tanımla
    3. LSTM eğitim ve test verilerini yükle
    4. Eğitim verisini train ve validasyon olarak ikiye ayıralım.
    5. LSTM model mimarisi oluşturma
    6. Modelin derlenmesi
    7. LSTM modeli eğitme
    8. Eğitilen modelin kaydedilmesi
    9. Eğitim geçmişini görselleştirme
    10. Test verisi üzerinden temel başarı sonucu
"""

# 1. Gerekli kütüphanelerin içeriye aktarılması
from pathlib import Path 
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split # eğitim verisini de train ve validasyon olarak ikiye ayıralım
from tensorflow.keras.layers import Dense, Dropout, LSTM # lstm model katmanları için kullanılacak katmanlar
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam 
from tensorflow.keras.utils import to_categorical # sayısal sınıf etiketlerini one-hot encoding formatına dönüştürmek

# 2. Proje klasör yollarını ve eğitim ayarlarını tanımla
PROJECT_DIR = Path(__file__).resolve().parent # çalışan python dosyanın bulunduğu klasörü proje ana dizini olarak alır
DATA_DIR  = PROJECT_DIR / "data" # ham ve işlenmiş veriler için ana veri klasörü
PROCESSED_DATA_DIR = DATA_DIR / "processed" # metadata gibi işlenmiş dosyaların kaydedileceği klasör 
MODELS_DIR = PROJECT_DIR / "models" # model dosyaları ve label encoder gibi nesnelerin kaydedileceği klasör
PLOTS_DIR = PROJECT_DIR / "outputs" / "plots" # grafik çıktıları klasörü

LSTM_X_TRAIN_PATH = PROCESSED_DATA_DIR / "lstm_X_train.npy" # lstm eğitim giriş verisi kaydedilecek yol
LSTM_X_TEST_PATH = PROCESSED_DATA_DIR / "lstm_X_test.npy" # lstm test giriş verisi kaydedilecek yol
Y_TRAIN_PATH = PROCESSED_DATA_DIR / "y_train.npy" # eğitim etiketlerinin kaydedileceği yol
Y_TEST_PATH = PROCESSED_DATA_DIR / "y_test.npy" # test etiketlerinin kaydedileceği yol

LSTM_MODEL_PATH = MODELS_DIR / "lstm_emotion_model.keras" # eğitilen lstm modelinin kaydedileceği yer
LSTM_ACCURACY_PLOT_PATH = PLOTS_DIR / "lstm_accuracy_history.png" # lstm accuracy grafiğinin kaydedileceği dosya yolu
LSTM_LOSS_PLOT_PATH = PLOTS_DIR / "lstm_loss_history.png" # # lstm kayıp grafiğinin kaydedileceği dosya yolu

BATCH_SIZE = 32 # her eğitim adımında modele verilecek örnek sayısı
EPOCHS = 30 # modelin eğitim verisi üzerinden maksimum kaç tur geçeceği
LEARNING_RATE = 0.0005 # Adam optimizasyon algoritmasının başlangıç öğrenme oranı
VALIDATION_SIZE = 0.2 # Eğitim verisinin %20 sinin validasyon olarak ayrılması
RANDOM_SEED = 42 # eğitim sürecinde ki tekrarlanabilirlik için

MODELS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

np.random.seed(RANDOM_SEED) 

# 3. LSTM eğitim ve test verilerini yükle
def load_training_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """LSTM eğitim ve test verilerini dosyadan yükler"""

    required_files = [LSTM_X_TRAIN_PATH, LSTM_X_TEST_PATH, Y_TRAIN_PATH, Y_TEST_PATH]

    for file_path in required_files: # gerekli dosyaların tamamını tek tek kontrol edelim
        if not file_path.exists():
            raise FileNotFoundError(f"Dosya bulunamadı: {file_path}")
        
    X_train = np.load(LSTM_X_TRAIN_PATH)
    X_test = np.load(LSTM_X_TEST_PATH)
    y_train = np.load(Y_TRAIN_PATH)
    y_test = np.load(Y_TEST_PATH)

    return X_train, X_test, y_train, y_test

#  4. Eğitim verisini train ve validasyon olarak ikiye ayıralım.
def split_train_validation(X_train: np.ndarray, y_train: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Eğitim verisini sınıf dağılımını koruyarak train/validasyon olarak ayırır"""

    X_train_main, X_val, y_train_main, y_val = train_test_split(X_train, y_train, test_size=VALIDATION_SIZE, random_state=RANDOM_SEED, stratify=y_train)

    return X_train_main, X_val, y_train_main, y_val

# 5. LSTM model mimarisi oluşturma
def build_lstm_model(input_shape: tuple[int, int], num_classes: int) -> Sequential:
    """MFCC zaman dizileri için LSTM model mimarisi oluştur"""

    model = Sequential() # LSTM katmanlarını sırayla eklemek için boş bir sequential model

    model.add(
        LSTM(
            units = 64, return_sequences = True, input_shape = input_shape
        )
    ) # ilk LSTM katmanı ile MFCC özelliklerinin zaman içindeki değişimlerini öğreniyoruz

    model.add(Dropout(0.3)) # overfitting engellemek için

    model.add(
        LSTM(
            units = 32, return_sequences = False
        )
    ) # zaman boyunca öğrenilen bilgiyi tek bir temsil katmanına dönüştür

    model.add(Dropout(0.3)) # ikinci LSTM sonrasında aşırı öğrenme riskini azaltıyoruz

    model.add(Dense(64, activation="relu")) # sınıflandırma
    model.add(Dropout(0.4))
    model.add(Dense(num_classes, activation="softmax")) # output katnanı

    return model

# 6. Modelin derlenmesi
def compile_model(model: Sequential) -> Sequential:
    """LSTM modelini eğitim için derler"""

    optimizer = Adam(learning_rate = LEARNING_RATE)

    model.compile(optimizer = optimizer, loss = "categorical_crossentropy", metrics = ["accuracy"])

    return model

# 7. LSTM modeli eğitme
def train_model(
        model: Sequential, 
        X_train_main: np.ndarray, 
        y_train_main_categorical: np.ndarray, 
        X_val: np.ndarray, 
        y_val_categorical: np.ndarray):
    """LSTM modelini train ve val verisi üzerinden eğitir"""

    history = model.fit(
        X_train_main, y_train_main_categorical,
        validation_data = (X_val, y_val_categorical),
        epochs = EPOCHS,
        batch_size = BATCH_SIZE,
        shuffle = True
    )

    return history

# 8. Eğitilen modelin kaydedilmesi
def save_model(model: Sequential) -> None:
    """Eğitilen LSTM modelini dosyaya kaydeder"""
    model.save(LSTM_MODEL_PATH)
    print("LSTM modeli kaydedildi.")

# 9. Eğitim geçmişini görselleştirme
def plot_training_history(history) -> None:
    """LSTM eğitim sürecindeki accuracy ve loss değerlerini görselleştirir"""

    plt.figure()
    plt.plot(history.history["accuracy"], label = "Eğitim Doğruluğu")
    plt.plot(history.history["val_accuracy"], label = "Validasyon Doğruluğu")
    plt.title("LSTM Eğitim Süreci - Doğruluk")
    plt.xlabel("Epoch")
    plt.ylabel("Doğruluk")
    plt.legend()
    plt.tight_layout()
    plt.savefig(LSTM_ACCURACY_PLOT_PATH)
    plt.close()

    plt.figure()
    plt.plot(history.history["loss"], label = "Eğitim Kaybı")
    plt.plot(history.history["val_loss"], label = "Validasyon Kaybı")
    plt.title("LSTM Eğitim Süreci - Kayıp")
    plt.xlabel("Epoch")
    plt.ylabel("Kayıp")
    plt.legend()
    plt.tight_layout()
    plt.savefig(LSTM_LOSS_PLOT_PATH)
    plt.close()

# 10. Test verisi üzerinden temel başarı sonucu
def evaluate_model(model: Sequential, X_test: np.ndarray, y_test_categorical: np.ndarray) -> None:
    """LSTM modelinin test verisi üzerindeki doğruluk ve kayıp değerlerini yazdır"""

    test_loss, test_accuracy = model.evaluate(X_test, y_test_categorical, verbose = 0)

    print(f"test_loss: {test_loss}")
    print(f"test_accuracy: {test_accuracy}")

def main() -> None:
    """LSTM model eğitim sürecini başlatır"""

    print("LSTM model eğitimi başladı...")

    X_train, X_test, y_train, y_test = load_training_data() # LSTM eğitim ve test verilerini dosyadan yükle
    print("Yüklenen veri boyutları")
    print(f"X_train: {X_train}")
    print(f"X_test: {X_test}")
    print(f"y_train: {y_train}")
    print(f"y_test: {y_test}")

    X_train_main, X_val, y_train_main, y_val = split_train_validation(X_train, y_train) # eğitim verisini train ve val olarak ayır
    print("Train validasyon ayrımı sonrası veri boyutları")
    print(f"X_train_main: {X_train_main}")
    print(f"X_val: {X_val}")
    print(f"y_train_main: {y_train_main}")
    print(f"y_val: {y_val}")

    num_classes = len(np.unique(y_train)) # eğitim etiketlerinden toplam sınıf sayısını hesapla
    input_shape = X_train.shape[1:]
    y_train_main_categorical = to_categorical(y_train_main, num_classes = num_classes)
    y_val_categorical = to_categorical(y_val, num_classes = num_classes)
    y_test_categorical = to_categorical(y_test, num_classes = num_classes)
    print("Model bilgileri")
    print(f"num_classes: {num_classes}")
    print(f"input_shape: {input_shape}")

    model = build_lstm_model(input_shape, num_classes) # LSTM mimarisi oluştur

    model = compile_model(model) # modeli loss, optimizer ve metric bilgileri ile eğitime hazır hale getir

    history = train_model(
        model=model,
        X_train_main=X_train_main,
        y_train_main_categorical=y_train_main_categorical,
        X_val=X_val,
        y_val_categorical=y_val_categorical
    ) # eğitim
 
    save_model(model) # eğitimi tamamlanam modelin kaydı
    plot_training_history(history) # eğitim sürecindeki accuracy ve loss değerlerini görselleştir
    evaluate_model(model, X_test, y_test_categorical) # temel seviye değerlendirme

if __name__ == "__main__":
    main()