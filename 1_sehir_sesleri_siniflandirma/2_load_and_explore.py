"""
Amaç:
    ESC-50 veri setinin etiket dosyasını oku ve veri setinin temel yapısını incele

Veri seti: 
    1. meta/esc50.csv dosyasını kullanalım
    2. csv dosyasının içerisinde ses dosyalarının isimleri, sınıfları, hedef etiketleri ve fold bilgileri
        - 1-155858-E-25.wav,1,25,footsteps,False,155858,E
        - 1-155858-F-25.wav,1,25,footsteps,False,155858,F

Adımlar:
    1. Gerekli kütüphanelerin içe aktarılması
    2. Veri seti klasör yollarının tanımlanması
    3. Metadata dosyasının okunması
    4. Veri setinin ilk satırlarının incelenmesi
    5. Veri seti boyutu ve sütunlarının incelenmesi
    6. Eksik değer kontrolünün yapılması
    7. Sınıf sayısı ve sınıf dağılımının incelenmesi
    8. Fold dağılımının incelenmesi
    9. Sınıf dağılım grafiğinin oluşturulması
    10. Örnek bir ses dosyasının kontrol edilmesi
"""

# 1. Gerekli kütüphanelerin içe aktarılması
import os 
import pandas as pd 
import matplotlib.pyplot as plt 

# 2. Veri seti klasör yollarının tanımlanması
DATASET_DIR = "data/raw/ESC-50-master"
META_DATA = os.path.join(DATASET_DIR, "meta", "esc50.csv") # etiket dosyasının yolu
AUDIO_DIR = os.path.join(DATASET_DIR, "audio") # wav ses dosyalarının bulunduğu klasör
OUTPUT_DIR = "outputs" # grafiklerin kaydedileceği klasör

# 3. Metadata dosyasının okunması
def load_metadata():
    """
        ESC-50 metadata dosyasını okur ve dataframe olarak döndürür
    """
    df = pd.read_csv(META_DATA) # esc50.csv dosyasını tablo formatında okur
    return df

# 4. Veri setinin ilk satırlarının incelenmesi
def show_first_rows(df):
    print(df.head())

# 5. Veri seti boyutu ve sütunlarının incelenmesi
def show_dataset_info(df):
    print(f"Veri seti boyutu: \n{df.shape}")
    print(df.columns.tolist())

# 6. Eksik değer kontrolünün yapılması
def check_missing_values(df):
    print(f"Eksik değer kontrolü: \n{df.isnull().sum()}") # her sütundaki eksik değer sayısı

# 7. Sınıf sayısı ve sınıf dağılımının incelenmesi
def show_class_distribution(df):
    """Sınıf sayısını ve her sınıftaki örnek sayısını gösterir"""
    print(f"Toplam sınıf sayısı: \n{df["category"].nunique()}") # benzersiz sınıf sayısını gösterir

    print(f"Her sınıftaki örnek sayısı: \n{df["category"].value_counts()}") # her sınıfta kaç örnek olduğunu gösterir

# 8. Fold dağılımının incelenmesi
def show_fold_distribution(df):
    """Veri setindeki fold dağılımını gösterir"""

    print(f"Fold dağılımı: \n{df["fold"].value_counts().sort_index()}") # her fold içinde ki örnek sayısını gösterir

# 9. Sınıf dağılım grafiğinin oluşturulması
def plot_class_distribution(df):
    """Sınıf dağılımını bar grafik olarak oluşturur ve kaydeder"""

    os.makedirs(OUTPUT_DIR, exist_ok=True) # outputs klasörünü yoksa oluştur
    class_counts = df["category"].value_counts() # her sınıftaki örnek sayısını hesapla

    plt.figure()
    class_counts.plot(kind = "bar") # sınıf dağılımını bir bar grafik olarak çizer
    plt.title("ESC-50 Sınıf Dağılımı")
    plt.xlabel("Ses Sınıfları")
    plt.ylabel("Örnek Sayısı")
    plt.xticks(rotation = 90)
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "class_distribution.png") # kayıt dosyası yolunu oluştur
    plt.savefig(save_path)
    plt.show()

# 10. Örnek bir ses dosyasının kontrol edilmesi
def chech_sample_audio_file(df):
    """ilk ses dosyasının klasörde bulunup bulunmadığını kontrol eder"""

    sample_filename = df.iloc[0]["filename"] # ilk ses dosyasının adını alır
    sample_category = df.iloc[0]["category"] # ilk ses dosyasının sınıf adını alır
    sample_path = os.path.join(AUDIO_DIR, sample_filename)

    if os.path.exists(sample_path):
        print("Dosya mevcut")
    else: print("Dosya bulunamadı")

df = load_metadata()
show_first_rows(df)
show_dataset_info(df)
check_missing_values(df)
show_class_distribution(df)
show_fold_distribution(df)
plot_class_distribution(df)
chech_sample_audio_file(df)