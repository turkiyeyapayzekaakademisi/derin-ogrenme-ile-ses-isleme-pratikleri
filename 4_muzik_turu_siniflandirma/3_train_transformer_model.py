"""
Amaç:
    - Veri setinden çıkarılan Mel Spectrogram özelliklerini kullanarak Conv1D destekli transformer tabanlı bir sınıflandırıcı inşa edelim
    - Eğitim sonunda en iyi modeli kaydet ve test verisi üzerinden değerlendir

Veri seti:
    - X_mel.npy, y.npy

Adımlar:
    1. Gerekli kütüphanelerin içeriye aktarılması
    2. Proje klasör yollarının tanımlanması
    3. Eğitim ayarlarının yapılması
    4. X ve y verilerinin yüklenmesi
    5. Verinin eğitim, validasyon ve test olarak ayrılması
    6. Transformer encoder bloğunun oluşturulması
    7. Conv1D destekli Transformer model mimarisinin oluşturulması
    8. Modelin derlenmesi
    9. Callback yapılarının tanımlanması
    10. Modelin eğitilmesi
    11. Test verisinin değerlemdirme dosyası için oluşturulması
"""
# 1. Gerekli kütüphanelerin içeriye aktarılması
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers # keras katmanlarını içeriye aktar
from tensorflow.keras import models # keras model nesnesini oluşturmak için
from tensorflow.keras import callbacks # Erken durdurma, model check point, learning rate azaltıcı
from sklearn.model_selection import train_test_split

# 2. Proje klasör yollarının tanımlanması
PROJECT_DIR = Path(__file__).resolve().parent # Bu python dosyasının bulunduğu klasörü proje ana klasörü olarak alır
PROCESSED_DIR = PROJECT_DIR / "data" / "processed" # işlenmiş dosya kaydedilecek klasör
MODELS_DIR = PROCESSED_DIR / "outputs" / "models" # eğitilmiş modelin kaydedileceği klasör
X_MEL_PATH = PROCESSED_DIR / "X_mel.npy" # mel-spectrogram özelliklerinin kaydedileceği dosya yolu
Y_PATH = PROCESSED_DIR / "y.npy" # sınıf etiketlerinin kaydedileceği dosya yolu
MODEL_PATH = MODELS_DIR / "transformer_music_genre_model.keras" # en iyi modelin kaydedileceği dosya yolu
X_TEST_PATH = PROCESSED_DIR / "X_test.npy" # test özellik kaydedilecek yol
Y_TEST_PATH = PROCESSED_DIR / "y_test.npy" # test etiketi kaydedilecek yol

MODELS_DIR.mkdir(parents=True, exist_ok=True)

# 3. Eğitim ayarlarının yapılması
RANDOM_STATE = 42 # tekrarlanabilir işler için sabit rastgelelik değeri
TEST_SIZE = 0.15 # veri setinin %15 nin test verisi olması
VALIDATION_SIZE = 0.15 # veri setinin 515 inin validasyon olması
BATCH_SIZE = 16 # eğitim sırasında her adımda modele verilecek örnek sayısı
EPOCHS = 30 # Modelin maksimum kaç epoch eğitileceğini tanımlar
LEARNING_RATE = 0.0002 # Adam için başlangıç öğrenme oranı

tf.keras.utils.set_random_seed(RANDOM_STATE) # tensorflow tarafında rastgeleliği sabitleyerek sonuçları daha tekrarlanabilir yapar

# 4. X ve y verilerinin yüklenmesi
X = np.load(X_MEL_PATH)
y = np.load(Y_PATH)
X = X.astype(np.float32)
y = y.astype(np.int64) 
input_shape = X.shape[1:] # (ses sayısı, h, w)
num_classes = len(np.unique(y)) # veri setindeki toplam sınıf sayısı

# 5. Verinin eğitim, validasyon ve test olarak ayrılması
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size = TEST_SIZE + VALIDATION_SIZE, random_state=RANDOM_STATE, stratify=y)

relative_test_size = TEST_SIZE/ (TEST_SIZE + VALIDATION_SIZE)

X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=relative_test_size, random_state=RANDOM_STATE, stratify=y_temp)

print(f"X_train shape: {X_train.shape}")
print(f"X_val shape: {X_val.shape}")
print(f"X_test shape: {X_test.shape}")

# 6. Transformer encoder bloğunun oluşturulması
def transformer_encoder_block(inputs, embed_dim, num_heads, ff_dim, dropout_rate):
    """Tek bir transformer encoder bloğu oluşturan fonksiyon tanımlar"""

    attention_output = layers.MultiHeadAttention(
        num_heads=num_heads, # kullanılacak attention başlığı sayısı
        key_dim = embed_dim // num_heads # her attention başlığının temsil boyutu
        )(inputs, inputs)
    
    attention_output = layers.Dropout(dropout_rate)(attention_output) # attention çıktısını dropout ekleyerek aşırı öğrenmeyi azaltır
    attention_output = layers.LayerNormalization(epsilon = 1e-6)(inputs + attention_output) # residual connection ve layer normalizasyon

    feed_forward_output = layers.Dense(ff_dim, activation="relu")(attention_output) # feed forward katmanıs
    feed_forward_output = layers.Dense(embed_dim)(feed_forward_output) # feed forward çıktısını tekrardan embedding boyutunai indir
    feed_forward_output = layers.Dropout(dropout_rate)(feed_forward_output)

    encoder_output = layers.LayerNormalization(epsilon=1e-6)(attention_output + feed_forward_output)

    return encoder_output

# 7. Conv1D destekli Transformer model mimarisinin oluşturulması
def build_transformer_model(input_shape, num_classes):
    """Conv destekli transformer model mimarisini oluşturur"""

    model_inputs = layers.Input(shape = input_shape) # modelin giriş katmanını mel spectrogram boyutuna göre oluştur

    x = layers.Conv1D(
        filters = 64, # filtre sayısı
        kernel_size = 5, # filtre boyutu yani kaç zaman adımına bakacağını belirliyoruz
        padding = "same", # zaman boyutunu korumak için 
        activation = "relu"  
    )(model_inputs) # zaman eksenindeki kısa süreli müzikal örüntüleri yakalamak için conv1d katmanı

    x = layers.BatchNormalization()(x) # normalizasyon
    x = layers.SpatialDropout1D(0.1)(x) # kanal bazlı dropout
    x = layers.MaxPooling1D(pool_size = 2)(x) # zaman boyutunu küçültmek ve hesaplama yükünü azaltmak için

    x = layers.Conv1D(filters = 96, kernel_size = 5, padding = "same", activation = "relu")(x)
    x = layers.BatchNormalization()(x) # normalizasyon
    x = layers.SpatialDropout1D(0.1)(x) # kanal bazlı dropout
    x = layers.Dense(96)(x) # conv çıktısını transformer block için embedding boyutuna dönüştür

    new_time_steps = x.shape[1] # max pooling sonrası kalan zaman adımı
    positions = tf.range(start = 0, limit = new_time_steps, delta = 1) # her zaman adımı için pozisyon indekslerini oluştur

    position_embedding = layers.Embedding(
        input_dim=new_time_steps, # pozisyon embedding için toplam zaman adımı sayısı
        output_dim=96 # pozisyon embedding boyutunu model embedding boyutu ile aynı yapar
    )(positions)

    x = x + position_embedding # zaman adımı temsilleri ile pozisyon bilgisini birleştir

    for _ in range(2): # belirlenen sayıda transformer encoder bloğu ekle
        x = transformer_encoder_block(inputs=x, embed_dim=96, num_heads=4, ff_dim=192, dropout_rate=0.15)

    x = layers.GlobalAveragePooling1D()(x) # zaman eksenindeki tüm temsilleri ortalayarak tek vektöre indirger
    x = layers.Dense(128, activation="relu")(x) # sınıflandırma öncesi daha güçlü temsil
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.15)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.15)(x)

    models_outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs = model_inputs, outputs = models_outputs)

    return model

# 8. Modelin derlenmesi
model = build_transformer_model(input_shape=input_shape, num_classes=num_classes) 

optimizer = tf.keras.optimizers.Adam(learning_rate = LEARNING_RATE)
model.compile(optimizer=optimizer, loss = "sparse_categorical_crossentropy", metrics = ["accuracy"])

# 9. Callback yapılarının tanımlanması
early_stopping = callbacks.EarlyStopping(
    monitor = "val_loss", # takip edilecek metrik
    patience=12, # kaç epoch iyileşme bekleriz
    restore_best_weights=True # eğitim sonunda en iyi ağırlıkların geri yüklenmesi
)

model_checkpoint = callbacks.ModelCheckpoint(
    filepath=MODEL_PATH, # en iyi modelin kaydedileceği dosya yolu
    monitor="val_loss", # en iyi modeli seçmek için val loss değerinin takip edilmesi
    save_best_only=True,
    verbose=1
)

reduce_lr = callbacks.ReduceLROnPlateau(
    monitor = "val_loss",
    factor = 0.5, # learning rate değerini yarıya düşürür
    patience = 5, # kaç epoch iyileşmes olmazsa learning rate azaltılacağını belirler
    min_lr = 1e-6,
    verbose = 1
)

callback_list = [early_stopping, model_checkpoint, reduce_lr]

# 10. Modelin eğitilmesi
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callback_list,
    verbose=1
)

# 11. Test verisinin değerlemdirme dosyası için oluşturulması
np.save(X_TEST_PATH, X_test)
np.save(Y_TEST_PATH, y_test)