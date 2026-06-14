"""
Amaç:
    - Eğitilmiş LSTM duygu sınıflandırma modelini test verisi üzerinden değerlendirelim.
    - classification report ve confusion matrix ile modelin hangi duygu sınıflarında başarılı olduğunu incelemek

Veri seti: lstm_x_test, y_test

Adımlar:
    1. Gerekli kütüphanelerin içeriye aktarılması
    2. Proje klasör yollarını tanımla
    3. Model, test verisini ve label encoder yükle 
    4. Test verisi üzerinden tahmin gerçekleştir
    5. Classification report çıktısını oluştur ve kaydet
    6. Confusion matrix görselini oluştur ve kaydet
"""

# 1. Gerekli kütüphanelerin içeriye aktarılması
from pathlib import Path 

import joblib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns # görselleştirme
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model # eğitilmiş lstm modelini yüklemek için

# 2. Proje klasör yollarını tanımla
PROJECT_DIR = Path(__file__).resolve().parent # çalışan python dosyanın bulunduğu klasörü proje ana dizini olarak alır
DATA_DIR  = PROJECT_DIR / "data" # ham ve işlenmiş veriler için ana veri klasörü
PROCESSED_DATA_DIR = DATA_DIR / "processed" # metadata gibi işlenmiş dosyaların kaydedileceği klasör 
MODELS_DIR = PROJECT_DIR / "models" # model dosyaları ve label encoder gibi nesnelerin kaydedileceği klasör
PLOTS_DIR = PROJECT_DIR / "outputs" / "plots" # grafik çıktıları klasörü
REPORTS_DIR = PROJECT_DIR / "outputs" / "reports" # metin raporlarının kaydedileceği klasörün tanımlanması

LSTM_MODEL_PATH = MODELS_DIR / "lstm_emotion_model.keras" # eğitilen lstm modelinin kaydedileceği yer
LABEL_ENCODER_PATH = MODELS_DIR / "label_encoder.pkl"

X_TEST_PATH = PROCESSED_DATA_DIR / "lstm_X_test.npy" # lstm test giriş verisi kaydedilecek yol
Y_TEST_PATH = PROCESSED_DATA_DIR / "y_test.npy" # test etiketlerinin kaydedileceği yol
REPORTS_PATH = REPORTS_DIR / "lstm_classification_report.txt" # classification report kayıt yolu
CONFUSION_MATRIX_PATH = PLOTS_DIR / "lstm_confusion_matrix.png" # confusion matrix görsel kayıt yolunu tanımlamak

PLOTS_DIR.mkdir(parents=True, exist_ok=True) 
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# 3. Model, test verisini ve label encoder yükle 
def load_required_files():
    """Model, test verisi ve label encoder dosyalarını yükler"""

    model = load_model(LSTM_MODEL_PATH) # eğitilmiş lstm modelini yükler
    label_encoder = joblib.load(LABEL_ENCODER_PATH) # sayısal sınıfları okunabilir duygu isimlerine çevirmek için encoder
    X_test = np.load(X_TEST_PATH)
    y_test = np.load(Y_TEST_PATH)

    return model, label_encoder, X_test, y_test 


# 4. Test verisi üzerinden tahmin gerçekleştir
def predict_classes(model, X_test: np.ndarray) -> np.ndarray:
    """Test verisi üzerinden sınıf tahminlerini üretir"""

    y_pred_probs = model.predict(X_test) # modelin her sınıf için olasılık tahminlerini alalım
    y_pred = np.argmax(y_pred_probs, axis=1) # en yüksek olasılığa sahip sınıfı tahmin sonucu olarak seçelim

    return y_pred

# 5. Classification report çıktısını oluştur ve kaydet
def save_classification_report(y_test: np.ndarray, y_pred: np.ndarray, class_names: list[str]) -> None:
    """Classification report (sınıflandırma raporu) oluşturur ve dosyayı kaydeder"""

    report = classification_report(y_test, y_pred, target_names=class_names) # precision, recall ve f1 score

    print("Classification Report")
    print(report)

    REPORTS_PATH.write_text(report, encoding="utf-8") # raporu metin dosyası olarak kaydet

# 6. Confusion matrix görselini oluştur ve kaydet
def save_confusion_matrix(y_test: np.ndarray, y_pred: np.ndarray, class_names: list[str]) -> None:
    """Confusion matrix görseli oluştur ve kaydet"""

    cm_matrix = confusion_matrix(y_test, y_pred)

    plt.figure()
    sns.heatmap(cm_matrix, annot= True, fmt = "d", cmap = "coolwarm", xticklabels=class_names, yticklabels=class_names)
    plt.title("LSTM Confusion Matrix")
    plt.xlabel("Tahmin edilen sınıf")
    plt.ylabel("Gerçek sınıf")
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH)
    plt.close()

def main():
    """LSTM model değerlendirme sürecini başlatır"""

    model, label_encoder, X_test, y_test = load_required_files() # gerekli dosyaların yüklenmesi
    class_names = list(label_encoder.classes_) # sayısal sınıflara karşılık gelen duygu isimlerini alalım

    y_pred = predict_classes(model, X_test) # test verisi üzerinden sınıf tahminleri yürütelim

    save_classification_report(y_test, y_pred, class_names)
    save_confusion_matrix(y_test, y_pred, class_names)

if __name__ == "__main__":
    main()