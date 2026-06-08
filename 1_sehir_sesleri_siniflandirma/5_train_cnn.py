"""
Amaç:
    Mel Spectrogram görüntülerini kullanarak CNN tabanlı bir ses sınıflandırma modeli eğitelim

Veri seti: data/processed/mel_spectrograms

Adımlar:
    1. Gerekli kütüphanelerin içeriye aktarılması
    2. Veri ve model klasör yollarının tanımlanması
    3. Eğitim ayarlarının tanımlanması
    4. Train ve validasyon veri setlerinin oluşturulması
    5. CNN model mimarisinin hazırlanması
    6. Eğitim grafiklerini kaydeden bir fonksiyon hazırla
    7. Ana çalışma akışı oluştur
    8. Modelin eğitilmesi
    9. Modelin ve sınıf isimlerinin kaydedilmesi
"""
# 1. Gerekli kütüphanelerin içeriye aktarılması
import os
import tensorflow as tf # derin öğrenme
from tensorflow.keras import layers, models # cnn katmanları ve model yapısı için kullanılır
import matplotlib.pyplot as plt

# 2. Veri ve model klasör yollarının tanımlanması
DATA_DIR = "data/processed/mel_spectrograms" # mel görsellerinin bulunduğu ana klasör
TRAIN_DIR = os.path.join(DATA_DIR, "train") # eğitim verisi
VAL_DIR = os.path.join(DATA_DIR, "val") # doğrulama verisi 
MODEL_DIR = "models" # eğitilen modelin kaydedileceği klasör
OUTPUT_DIR = "outputs"

# 3. Eğitim ayarlarının tanımlanması
IMG_SIZE = (128, 128) # modele verilecek görsellerin boyutu
BATCH_SIZE = 32 # modelin aynı anda işleyebileceği örnek sayısı
EPOCHS = 250 # Eğitim tur sayısı
LEARNING_RATE = 0.0001 # öğrenme oranı

# 4. Train ve validasyon veri setlerinin oluşturulması
def create_datasets():
    """Train ve validasyon görsellerini tensorflow veri seti olarak hazırla"""
    train_ds = tf.keras.utils.image_dataset_from_directory( # eğitim görsellerini klasör isimlerine göre veri setine çevirir
        TRAIN_DIR,
        image_size = IMG_SIZE,
        batch_size = BATCH_SIZE,
        label_mode = "categorical"
    )

    val_ds = tf.keras.utils.image_dataset_from_directory( # Validasyon görsellerini klasör isimlerine göre veri setine çevirir
        VAL_DIR,
        image_size = IMG_SIZE,
        batch_size = BATCH_SIZE,
        label_mode = "categorical"
    )

    class_names = train_ds.class_names # sınıf isimlerini klasör adlarından alır
    AUTOTUNE = tf.data.AUTOTUNE # veri okuma performansı için
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size = AUTOTUNE) # eğitim verisini hızlı ve karışık şekilde oku
    val_ds = val_ds.cache().prefetch(buffer_size = AUTOTUNE)    

    return train_ds, val_ds, class_names

# train_ds, val_ds, class_names = create_datasets()

# 5. CNN model mimarisinin hazırlanması
def build_cnn_model(num_classes):
    """Mel spectrogram görüntülerini sınıflandıracak CNN modelini oluştur"""

    model = models.Sequential(
        [
            layers.Input(shape = (IMG_SIZE[0], IMG_SIZE[1], 3)), # modelin giriş görüntü boyutu (128, 128, 3)
            layers.Rescaling(1.0/255), # görüntülerde bulunan piksel değerlerinin 0-1 arasına sıkıştırılması

            layers.Conv2D(32, (3,3), activation="relu", padding="same"), # ilk temel görsel özellikleri çıkarır
            layers.MaxPooling2D((2,2)), # görsel boyutu küçültür

            layers.Conv2D(64, (3,3), activation="relu", padding = "same"), # daha belirgin spectrogram örüntülerini öğrenir
            layers.MaxPooling2D((2,2)), # özellik haritasını küçültür

            layers.Conv2D(128, (3,3), activation="relu", padding = "same"), # daha karmaşık ses-görsel özellikleri öğrenilir
            layers.MaxPooling2D((2,2)), # hesaplama yükünün azalması

            layers.Conv2D(256, (3,3), activation="relu", padding="same"), # üst seviye özellik çıkarımı
            layers.GlobalAveragePooling2D(), # Flatten yerine daha az parametreli özet özellik çıkarıcı

            layers.Dense(128, activation="relu"),
            layers.Dropout(0.5), # overfitting i azaltmak için
            layers.Dense(num_classes, activation="softmax") # 50 sınıf için her bir sınıfın olma olasılığını hesaplar

        ]
    )

    model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate = LEARNING_RATE),
        loss = "categorical_crossentropy",
        metrics = ["accuracy"]
    )

    return model

# model = build_cnn_model(50)

# 6. Eğitim grafiklerini kaydeden bir fonksiyon hazırla
def plot_training_history(history):
    """Eğitim ve validasyon accuracy/loss grafiklerini oluşturur"""

    os.makedirs(OUTPUT_DIR, exist_ok=True) # outputs klasörü yoksa oluşturur

    plt.figure()
    plt.plot(history.history["accuracy"], label = "Train Accuracy") # eğitim accuracy değerlerini çizdirir
    plt.plot(history.history["val_accuracy"], label = "Validation Accuracy") # validasyon accuracy değerini çizer
    plt.title("Model Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "training_accuracy.png"))
    plt.show()

    plt.figure()
    plt.plot(history.history["loss"], label = "Train Loss") # eğitim loss değerlerini çizdirir
    plt.plot(history.history["val_loss"], label = "Validation Loss") # validasyon loss değerini çizer
    plt.title("Model Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "training_loss.png"))
    plt.show()


# 7. Ana çalışma akışı oluştur
def main():
    """CNN model eğitim adımlarını sırasıyla çalıştırmak"""

    os.makedirs(MODEL_DIR, exist_ok=True)

    train_ds, val_ds, class_names = create_datasets() # train ve validasyon veri setlerini oluşturur
    num_classes = len(class_names) # toplam sınıf sayısını hesaplar

    model = build_cnn_model(num_classes) # CNN modelini oluştur
    model.summary() # model mimarisini ekrana yazdır

    checkpoint_path = os.path.join(MODEL_DIR, "best_esc50_cnn_model.keras") # en iyi modelin kayıt yolu
    checkpoint = tf.keras.callbacks.ModelCheckpoint( # validasyon accuracy nin en iyi olduğu modeli kaydeder
        checkpoint_path,
        monitor="val_accuracy",
        save_best_only=True,
        mode = "max",
        verbose=1
    )

    early_stop = tf.keras.callbacks.EarlyStopping( # erken durdurma
        monitor="val_loss",
        patience=7,
        restore_best_weights=True
    )

    # 8. Modelin eğitilmesi
    history = model.fit(
        train_ds,
        validation_data = val_ds,
        epochs = EPOCHS,
        callbacks = [checkpoint, early_stop]
    ) 

    # 9. Modelin ve sınıf isimlerinin kaydedilmesi
    final_model_path = os.path.join(MODEL_DIR, "final_esc50_cnn_model.keras") # final modelin kayıt yolu
    model.save(final_model_path)

    class_names_path = os.path.join(MODEL_DIR, "class_names.txt")
    with open(class_names_path, "w", encoding="utf-8") as file: # sınıf isimleri dosyasını yazma modunda aç
        for class_name in class_names:
            file.write(class_name + "\n") # her sınıf adını ayrı ayır satıra yazar

    plot_training_history(history) # eğitim sürecine ait grafikleri oluşturur

    print("Eğitim tamamlandı")

if __name__ == "__main__":
    main()
