"""
Amaç:
    - Hazırlanan gürültülü ve temiz ses verilerini kullanarak Conv1D tabanlı bir autoencoder modeli eğitilen.
    - Eğitilen model, gürültülü seslerden temiz ses üretmeyi öğrenir.

Veri seti:
    - data/processed: x_train, x_test, y_train, y_test

Adımlar:
    1. Gerekli kütüphanelerin içeriye aktarılması
    2. Proje klasör yollarının ve eğitim ayarlarının tanımlanması
    3. Gerekli klasörlerin oluşturulması
    4. Eğitim ve test verilerinin yüklenmesi
    5. Autoencoder model mimarisinin oluşturulması
    6. Modelin compile edilmesi
    7. Model callback yapılarının tanımlanması
    8. Modelin Eğitilmesi
    9. Eğitim geçmişinin görselleştirilmesi
    10. Test verisi üzerinden temel değerlendirme yapılması
    11. Ana çalışma akışının çalıştırılması
"""

# 1. Gerekli kütüphanelerin içeriye aktarılması
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model # Keras model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, UpSampling1D # autoencoder için gerekli layer lar
from tensorflow.keras.optimizers import Adam 
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau # eğitim sürecini kontrol etmek için kullanılacak callback ler

# 2. Proje klasör yollarının ve eğitim ayarlarının tanımlanması
BASE_DIR = Path(__file__).resolve().parent # Bu python dosyasının bulunduğu klasörü proje ana klasörü olarak alır
DATA_DIR = BASE_DIR / "data" # tüm veri dosyaları için ana klasör
PROCESSED_DATA_DIR = DATA_DIR / "processed" # model eğitime hazır numpy dosyalarının kaydedileceği klasör

MODELS_DIR = BASE_DIR / "models" # eğitilen model dosyalarının kaydedileceği klasör
OUTPUTS_DIR = BASE_DIR / "outputs" # Eğitim çıktılarımızın kaydedileceği ana klasör
FIGURES_DIR = OUTPUTS_DIR / "figures" # eğitim grafiklerinin kaydedileceği klasör

MODELS_PATH = MODELS_DIR / "audio_denoising_autoencoder.keras" # eğitilen modelin kaydedileceği dosya yolu
HISTORY_FIGURE_PATH = FIGURES_DIR / "training_history.png" # eğitim grafiğinin kaydedileceği dosya yolu

BATCH_SIZE = 32 # modelin her eğitim adımında kaç örnek kullanacağı
EPOCHS = 30 # modelin eğitim verisi üzerinden kaç tur atacağını belirle
LEARNING_RATE = 0.001 # Adam optimizasyon algoritmasının öğrenme oranı
VALIDATION_SPLIT = 0.1 # eğitim verisinin yüzde kaçının calidasyon için ayrılacağı

# 3. Gerekli klasörlerin oluşturulması
MODELS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# 4. Eğitim ve test verilerinin yüklenmesi
def load_datasets():
    """Eğitim ve test verilerini yükle"""
    X_train = np.load(PROCESSED_DATA_DIR / "X_train.npy")
    X_test = np.load(PROCESSED_DATA_DIR / "X_test.npy")
    y_train = np.load(PROCESSED_DATA_DIR / "y_train.npy")
    y_test = np.load(PROCESSED_DATA_DIR / "y_test.npy")

    return X_train, X_test, y_train, y_test

# 5. Autoencoder model mimarisinin oluşturulması
def build_autoencoder(input_shape):
    """Conv1D tabanlı autoencoder modelini oluşturur"""

    input_layer = Input(shape = input_shape) # modelin gürültülü ses giriş katmanı

    x = Conv1D(filters = 32, kernel_size = 9, activation = "relu", padding = "same")(input_layer) # giriş sesinden düşük seviyeli özellikleri çıkartır
    x = MaxPooling1D(pool_size = 2, padding = "same")(x) # zaman boyutunu yarıya indirerek sıkıştırma işlemi yapar

    x = Conv1D(filters = 64, kernel_size = 9, activation = "relu", padding = "same")(x) # daha yüksek seviyeli ses özellikleri çıkartır
    x = MaxPooling1D(pool_size = 2, padding = "same")(x) # zaman boyutunu tekrar yarıya indirir

    x = Conv1D(filters = 128, kernel_size = 9, activation = "relu", padding = "same")(x) # daha yoğun temsil öğrenir
    encoded = MaxPooling1D(pool_size = 2, padding = "same")(x) 

    x = Conv1D(filters = 128, kernel_size = 9, activation = "relu", padding = "same")(encoded) # sıkıştırılmış temsilden yeniden yapılandırma
    x = UpSampling1D(size = 2)(x) # zaman boyutunu iki katına çıkarır

    x = Conv1D(filters = 64, kernel_size = 9, activation= "relu", padding = "same")(x)
    x = UpSampling1D(size = 2)(x)

    x = Conv1D(filters = 32, kernel_size = 9, activation = "relu", padding = "same")(x)
    x = UpSampling1D(size = 2)(x)

    output_layer = Conv1D(filters = 1, kernel_size = 9, activation = "tanh", padding = "same")(x) # temizlenmiş ses sinyalini tek kanallı çıktı olarak üretir.

    model = Model(inputs = input_layer, outputs = output_layer) # giriş ve çıkış katmanlarını kullanarak autoencoder modeli oluştu
    return model

# 6. Modelin compile edilmesi
def compile_model(model):

    optimizer = Adam(learning_rate = LEARNING_RATE)
    model.compile(optimizer = optimizer, loss = "mse", metrics = ["mae"]) # modeli mse kayıp fonksiyonu ve mae metriği ile derle

    return model

# 7. Model callback yapılarının tanımlanması
def create_callbacks(): 
    """Eğitim sürecinde kullanılacak olan callback ler tanımlanır"""

    checkpoint = ModelCheckpoint( # en iyi modeli otomatik kaydetmek için
        filepath=MODELS_PATH, # modelin kaydedileceği dosya yolunu belirler
        monitor = "val_loss", # en iyi modeli seçmek için validasyon kaybını takip eder
        save_best_only = True, # sadece en iyi validasyon kaybına sahip modeli kaydeder
        verbose = 1
    )

    early_stopping = EarlyStopping( # eğitim gereksiz yere uzarsa durdurmak için early stopping kullan
        monitor="val_loss", # durdurma kararını validasyon kaybına göre verelim
        patience=5, # validasyon kaybı 5 epoch boyunca iyileşmezse eğitimi durdur
        restore_best_weights=True, # eğitim sonunda en iyi ağırlıkları geri yükler
        verbose=1
    )

    reduce_lr = ReduceLROnPlateau( # öğrenme oranını otomatik azaltmak için callback yapısını oluşturur
        monitor = "val_loss", # öğrenme oranı kararında validasyon loss takip eder
        factor=0.5, # öğrenme oranını yarıya indirir
        patience=3, # validasyon kaybı 3 epoch boyunca iyileşmez ise lr azaltılır
        min_lr = 1e-6, # öğrenme oranının düşebileceği en küçük değer
        verbose=1
    )

    callbacks = [checkpoint, early_stopping, reduce_lr]

    return callbacks

# 8. Modelin Eğitilmesi
def train_model(model, X_train, y_train, callbacks):
    """Autoencoder modelini eğiten fonksiyonu tanımlar"""

    history = model.fit(
        X_train, # gürültülü sesler model girdisi
        y_train, # modelin hedefi yani temiz sesler
        batch_size = BATCH_SIZE, # her eğitim adımında kullanılacak örnek sayısı
        epochs = EPOCHS, # toplam eğitim turu sayısı
        validation_split = VALIDATION_SPLIT, # eğitim verisinin bir kısmını validasyon olarak ayıralım
        callbacks = callbacks, # eğitim sürecinin callback ler ile kontrol edilmesi
        shuffle = True # eğitim verisi her epoch başında karıştırılır
    )

    return history

# 9. Eğitim geçmişinin görselleştirilmesi
def plot_training_history(history):
    """Eğitim ve validasyon kayıplarını görselleştir"""

    plt.figure()

    plt.plot(history.history["loss"], label = "Eğitim Loss")
    plt.plot(history.history["val_loss"], label = "Validasyon Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Autoencoder Eğitim Geçmişi")
    plt.legend()
    plt.grid(True)
    plt.savefig(HISTORY_FIGURE_PATH)
    plt.show()

# 10. Test verisi üzerinden temel değerlendirme yapılması
def evaluate_model(model, X_test, y_test):
    """Eğitilen modeli test verisi üzerinden değerlendirir"""

    test_loss, test_mae = model.evaluate(X_test, y_test, verbose = 1) # modelin gürültülü veri ve temiz hedefler arasında değerlendirilmesi
    print(f"test_loss: {test_loss}")
    print(f"test_mae: {test_mae}")

# 11. Ana çalışma akışının çalıştırılması
def main():
    """Dosyanın ana çalışma fonksiyonu"""

    X_train, X_test, y_train, y_test = load_datasets() # eğitim ve test verilerini yükler
    input_shape = X_train.shape[1:] # model giriş boyutunu eğitim verisinin şekline göre belirler

    model = build_autoencoder(input_shape) # autoencoder model mimarisi oluştur
    model = compile_model(model) # derleme
    callbacks = create_callbacks() # eğitim callback yapılarını oluştur
    history = train_model(model, X_train, y_train, callbacks) # modelin eğitimi

    plot_training_history(history)
    evaluate_model(model, X_test, y_test) # eğitilen modeli test verisi üzerinde değerlendir
    model.save(MODELS_PATH)

if __name__ == "__main__":
    main()

