"""
Amaç:
    Eğitilen CNN modelini test verisi üzerinde değerlendir ve başarı metrikleri ile incele.

Veri seti: test veri setini kullan

Adımlar:
    1. Gerekli kütüphanelerin içeriye aktarılması
    2. Veri, model ve çıktı klasör yollarının tanımlanması
    3. Test ayarlarının tanımlanması
    4. Sınıf isimlerinin dosyadan okunması
    5. Test veri setini oluştur
    6. Eğitilmiş modelin yüklenmesi
    7. Test verisi üzerinde tahmin yapılması
    8. Başarı metriklerinin hesaplanması
    9. Confusion matrix grafiğinin oluşturulması
    10. Ana çalışma akışının oluşturulması
"""
# 1. Gerekli kütüphanelerin içeriye aktarılması
import os
import numpy as np
import pandas as pd
import tensorflow as tf 
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 2. Veri, model ve çıktı klasör yollarının tanımlanması
DATA_DIR = "data/processed/mel_spectrograms" # mel görsellerinin bulunduğu ana klasör
TEST_DIR = os.path.join(DATA_DIR, "test") # test verisi
MODEL_PATH = "models/best_esc50_cnn_model.keras" # test edilecek en iyi modelin yolu
CLASS_NAME_PATH = "models/class_names.txt"
OUTPUT_DIR = "outputs"

# 3. Test ayarlarının tanımlanması
IMG_SIZE = (128, 128) # modele verilecek görsellerin boyutu
BATCH_SIZE = 32 # modelin aynı anda işleyebileceği örnek sayısı

# 4. Sınıf isimlerinin dosyadan okunması
def load_class_names():
    """Eğitim sırasında kaydedilen sınıf isimlerinin dosyadan okunması"""
    with open(CLASS_NAME_PATH, "r", encoding="utf-8") as file:
        class_names = [line.strip() for line in file.readlines()]

    return class_names

# print(load_class_names())

# 5. Test veri setini oluştur
def create_test_dataset():
    """test görsellerini tensorflow veri seti olarak hazırlar."""

    test_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        image_size = IMG_SIZE,
        batch_size = BATCH_SIZE,
        label_mode = "categorical",
        shuffle = False
    )

    return test_ds

# 6. Eğitilmiş modelin yüklenmesi
def load_trained_model():
    """Kaydedilmiş modelin yüklenmesi"""

    model = tf.keras.models.load_model(MODEL_PATH) # eğitilmiş modeli .keras dosyasından yükler

    return model

# 7. Test verisi üzerinde tahmin yapılması
def predict_test_data(model, test_ds):
    """Test veri seti üzerinden tahmin yapar ve gerçek/tahmin etiketleri döndürür."""
    y_true = [] # gerçek sınıf indeksleri
    y_pred = [] # tahmin edilen sınıf indekslerini tutar
    y_confidence = [] # tahmin güven skorları

    for images, labels in test_ds: # test veri setinde ki batchleri sırasıyla gezer
        predictions = model.predict(images, verbose = 0) # modelin sınıf olasılıklarını üretmesini sağlar
        true_classes = np.argmax(labels.numpy(), axis = 1) 
        predicted_classes = np.argmax(predictions, axis = 1) # en yüksek olasılıklı sınıfı tahmin olarak alır
        confidence_scores = np.max(predictions, axis=1) # her tahmin için en yüksek olasılık değeri

        y_true.extend(true_classes) # gerçek sınıf indekslerini listeye ekler
        y_pred.extend(predicted_classes) # tahmin edilen sınıf indekslerini listeye ekler
        y_confidence.extend(confidence_scores) # güven skorlarını listeye ekler

    return y_true, y_pred, y_confidence


# 8. Başarı metriklerinin hesaplanması
def calculate_metrics(y_true, y_pred, class_names):
    """Accuracy ve classification report sonuçlarını hesaplar"""

    accuracy = accuracy_score(y_true, y_pred) # doğruluk
    print(f"Accuracy: \n{accuracy}")

    clf_report = classification_report(y_true, y_pred, target_names=class_names) # precision, recall, f1 score ve accuracy
    print(f"Classification report: \n{clf_report}")

# 9. Confusion matrix grafiğinin oluşturulması
def save_confusion_matrix(y_true, y_pred, class_names):
    """Confusion matrix grafiği oluştur ve kaydet"""

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred) # gerçek ve tahmin sınıflarına göre confusion matrix hesapla

    plt.figure()
    plt.imshow(cm, interpolation="nearest", cmap = "coolwarm")
    plt.title("Confusion Matrix")
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation = 90)
    plt.yticks(tick_marks, class_names)

    plt.xlabel("Tahmin edilen sınıf")
    plt.ylabel("Gerçek sınıf")
    plt.tight_layout()

    save_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    plt.savefig(save_path)
    plt.show()

# 10. Ana çalışma akışının oluşturulması
def main():
    """Model test adımlarını sırasıyla çalıştırır"""

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(MODEL_PATH): # model dosyasının var olup olmadığını kontrol eder
        print("Model dosyası bulunamadı")
        return
    
    if not os.path.exists(TEST_DIR): # Test klasörü var olup olmadığını kontrol eder
        print("Test klasörü bulunamadı")
        return 
    
    class_names = load_class_names() # sınıf isimlerini dosyadan okur
    test_ds = create_test_dataset() # test veri setini oluşturur
    model = load_trained_model() # eğitilmiş modeli yükler

    y_true, y_pred, y_confidence = predict_test_data(model, test_ds) # test verisi üzerinden tahmin yapar
    calculate_metrics(y_true, y_pred, class_names) # test başarı metriklerini hesaplar
    save_confusion_matrix(y_true, y_pred, class_names) # confusion matrix oluştur ve kaydet

if __name__ == "__main__":
    main()
    
