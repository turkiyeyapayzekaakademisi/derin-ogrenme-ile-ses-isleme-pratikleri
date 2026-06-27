"""
Amaç:
    - Eğitilen Conv1D destekli transformer modeli test verisi üzerinde değerlendirilir.
    - Classification report ve confusion matrix sonuçları terminale yazdırılır.

Veri seti:
    - X_test ve y_test verisi ile test yapılır

Adımlar:
    1. Gerekli kütüphanelerin içeriye aktarılması
    2. Proje klasör yollarının tanımlanması
    3. Test verilerinin yüklenmesi
    4. Sınıf isimlerinin tanımlanması
    5. Eğitilmiş modelin yüklenmesi
    6. Modelin test verisi üzerinde tahmin yapması
    7. Classification report çıktısı oluşturulması
    8. Confusion matrix çıktısının oluşturulması
    9. Değerlendirme sonuçlarının incelenmesi
"""
# 1. Gerekli kütüphanelerin içeriye aktarılması
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

# 2. Proje klasör yollarının tanımlanması
PROJECT_DIR = Path(__file__).resolve().parent # Bu python dosyasının bulunduğu klasörü proje ana klasörü olarak alır
PROCESSED_DIR = PROJECT_DIR / "data" / "processed" # işlenmiş dosya kaydedilecek klasör
MODELS_DIR = PROCESSED_DIR / "outputs" / "models" # eğitilmiş modelin kaydedileceği klasör
MODEL_PATH = MODELS_DIR / "transformer_music_genre_model.keras" # en iyi modelin kaydedileceği dosya yolu

X_TEST_PATH = PROCESSED_DIR / "X_test.npy" # test özellik kaydedilecek yol
Y_TEST_PATH = PROCESSED_DIR / "y_test.npy" # test etiketi kaydedilecek yol

# 3. Test verilerinin yüklenmesi
X_test = np.load(X_TEST_PATH) # test özellik verisini numpy dizisi olarak yükle
y_test = np.load(Y_TEST_PATH) # test sınıf etiketlerini numpy dizisi olarak yükle

X_test = X_test.astype(np.float32) # float32 formatına dönüştürme
y_test = y_test.astype(np.int64) # test etiketlerini tam sayı formatına dönüştür

# 4. Sınıf isimlerinin tanımlanması
label_names = [
    "blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"
]

# 5. Eğitilmiş modelin yüklenmesi
model = tf.keras.models.load_model(MODEL_PATH) # kaydedilen/eğitilen modeli yükle

# 6. Modelin test verisi üzerinde tahmin yapması
y_pred_probabilities = model.predict(X_test) # her test örneği için sınıf olasılıklarını döndürür

y_pred = np.argmax(y_pred_probabilities, axis = 1) # en yüksek olasılığa sahip sınıfı tahmin etiketi olarak alır

# 7. Classification report çıktısı oluşturulması
report_text = classification_report(y_test, y_pred, target_names=label_names)

# 8. Confusion matrix çıktısının oluşturulması
confusion_matrix_values = confusion_matrix(y_test, y_pred)
confusion_matrix_df = pd.DataFrame(confusion_matrix_values, index = label_names, columns=label_names)

# 9. Değerlendirme sonuçlarının incelenmesi
print("Classification report")
print(report_text)

print("Confusion Matrix")
print(confusion_matrix_df)