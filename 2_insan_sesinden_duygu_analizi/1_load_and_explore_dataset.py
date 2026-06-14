"""
Amaç:
    RAVDESS veri setini okuayarak ses dosyalarını, duygu etiketlerini ve temel veri özelliklerinin analiz edilmesi.
    Oluşturulan metadata dosyasını sonraki aşamalarda MFCC çıkarımı ve LSTM model eğitimi için temel veri kaynağı olarak kullan.

Veri seti: https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio
    RAVDESS, insan sesinden duygu analizi çalışmaların kullanılan etiketli bir veri seti.
    Veri setinde 24 farklı insana ait konuşma kayıtları var, toplamda 1440 adet konuşma var. 
    Duygu sınıfları 8 kategoriden oluşuyor: neutral, calm, happy, sad, angry, fearful, disgust ve surprised
    Ses kayıtları birkaç saniye uzunluğunda ve hepsi .wav formatında
    Dosya adları duygu sınıfı, duygu yoğunluğu, konuşmacı numarası ve tekrar bilgisi gibi etiketler ile kodlanmış durumda

Adımlar:
    1. Gerekli kütüphanelerin içeriye aktarılması
    2. Proje klasör yollarının tanımlanması
    3. RAVDESS duygu etiketlerini sınıf isimleriyle eşleştir
    4. Veri setindeki tüm .wav dosyalarını tarayalım
    5. Dosya adlarından duygu sınıfı ve aktör bilgileri çıkart.
    6. Metadata tablosu oluştur
    7. Metadata tablosunu csv olarak kaydet
    8. Veri setinin genel özetini incele
    9. Duygu sınıflarının dağılımı görselleştir
    10. Örnek bir ses dosyasının waveform grafiğini oluştur
"""

# 1. Gerekli kütüphanelerin içeriye aktarılması
from pathlib import Path # Dosya ve klasör yollarını yönetmek için
import librosa # ses dosyalarını okumak ve ses sinyali özelliklerini analiz etmek
import librosa.display # waveform görselleştirme
import matplotlib.pyplot as plt
import pandas as pd

# 2. Proje klasör yollarının tanımlanması
PROJECT_DIR = Path(__file__).resolve().parent # çalışan python dosyanın bulunduğu klasörü proje ana dizini olarak alır
DATA_DIR  = PROJECT_DIR / "data" # ham ve işlenmiş veriler için ana veri klasörü
RAW_DATA_DIR = DATA_DIR / "raw" / "ravdess" # RAVDESS ses dosyaları
PROCESSED_DATA_DIR = DATA_DIR / "processed" # metadata gibi işlenmiş dosyaların kaydedileceği klasör 
PLOTS_DIR = PROJECT_DIR / "outputs" / "plots" # grafik çıktıları klasörü

PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True) # processed klasörü yoksa oluştur
PLOTS_DIR.mkdir(parents=True, exist_ok=True) # Grafik çıktı klasörü yoksa oluştur

# 3. RAVDESS duygu etiketlerini sınıf isimleriyle eşleştir
EMOTION_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
} # dosya adında ki sayısal duygu kodlarını okunabilir duygu isimlerine dönüştürelim

#  4. Veri setindeki tüm .wav dosyalarını tarayalım
def find_audio_files(dataset_dir: Path) -> list[Path]:
    """Veri seti klasörü içerisinde ki tüm wav dosyalarının bulunması"""

    audio_files = sorted(dataset_dir.rglob("*.wav")) # alt klasörler dahil tüm wav dosyalarını sıralı şekilde bulur
    return audio_files # bulunan dosya yollarını sonraki işlemlerde kullanmak için döndür

# print(find_audio_files(RAW_DATA_DIR))

# 5. Dosya adlarından duygu sınıfı ve aktör bilgileri çıkart.
def parse_ravdess_filename(file_path: Path) -> dict:
    """RAVDESS dosya adından etiket ve konusmacı bilgilerini çıkarır"""

    file_name = file_path.stem # dosya uzantısını kaldırarak sadece isim bölümünü alalım # 03-01-01-01-01-01-01
    parts = file_name.split("-")

    if len(parts) != 7:
        raise ValueError("Beklenmeyen bir dosya adı")
    
    modality = parts[0] # veri türü bilgisi
    vocal_channel = parts[1] # kaydın konuşma kanalı bilgis
    emotion_code = parts[2] # duygu sınıfı sayısal kodu
    intensity = parts[3] # duygunun normal yada güçlü yoğunlukta olup olmadığını anlamak için
    statement = parts[4] # ses kaydında kullanılan cümle tipi
    repetition = parts[5] # aynı ifadenin kaçıncı tekrarı olduğu
    actor = parts[6] # ses kaydını oluşturan aktör numarası

    emotion_label = EMOTION_MAP.get(emotion_code, "unknown") # duygu kodunu insan tarafından okunabilir hale çevir

    return {
        "file_path": str(file_path),
        "file_name": file_path.name,
        "modality": modality,
        "vocal_channel": vocal_channel,
        "emotion_code": emotion_code, 
        "emotion": emotion_label,
        "intensity": intensity,
        "statement": statement,
        "repetition": repetition,
        "actor":actor
    }
    
# 6. Metadata tablosu oluştur
def create_metadata(audio_files: list[Path]) -> pd.DataFrame:
    """Ses dosyalarından düzenli bir metadata tablosu oluştur"""
    records = [] # her ses dosyasından çıkan bilgileri bu listede topla

    for file_path in audio_files: # bulunan tüm ses dosyalarını tek tek işle
        try:
            record = parse_ravdess_filename(file_path) # dosya adından duygu gibi etiket bilgileri çıkartır
            records.append(record)
        except Exception as e:
            print(e)

    metadata_df = pd.DataFrame(records)
    return metadata_df

# 7. Metadata tablosunu csv olarak kaydet
def save_metadata(metadata_df: pd.DataFrame, save_path: Path) -> None:
    """Metadata tablosunu csv dosyası olarak kaydeder"""

    metadata_df.to_csv(save_path, index = False, encoding="utf-8")

# 8. Veri setinin genel özetini incele
def print_dataset_summary(metadata_df: pd.DataFrame) -> None:
    """Veri setine ait temel sayısal özeti ekrana yazdır"""

    total_samples = len(metadata_df) # veri setinde ki toplam ses dosyası sayısı
    total_classes = metadata_df["emotion"].nunique() # veri setinde ki benzersiz duygu sınıfı sayısı
    total_actors = metadata_df["actor"].nunique() # Veri setinde ki benzersiz konuşmacı sayısı

    print(f"total_samples: {total_samples}")
    print(f"total_classes: {total_classes}")
    print(f"total_actors: {total_actors}")

    print(f"Duygu sınıfı dağılımı: \n{metadata_df["emotion"].value_counts()}")

# 9. Duygu sınıflarının dağılımı görselleştir
def plot_emotiın_distribution(metadata_df: pd.DataFrame, save_path:Path) -> None:

    emotion_counts = metadata_df["emotion"].value_counts().sort_index() 

    plt.figure()
    emotion_counts.plot(kind = "bar")
    plt.title("Duygu sınıfı dağılımı")
    plt.xlabel("Duygu sınıfı")
    plt.ylabel("Ses dosyası sayısı")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# 10. Örnek bir ses dosyasının waveform grafiğini oluştur
def plot_sample_waveform(metadata_df: pd.DataFrame, save_path: Path) -> None:
    """Örnek bir ses dosyasının waveform grafiğini kaydeder"""

    sample_row = metadata_df.iloc[0] # veri setinden ilk ses kaydını örnek olarak seç
    sample_file_path = sample_row["file_path"] # seçilen örneğin dosya yolu
    sample_emotion = sample_row["emotion"] # duygu etiketi

    signal, sample_rate = librosa.load(sample_file_path, sr = None) # Ses dosyasını yükle

    duration = librosa.get_duration(y = signal, sr = sample_rate)

    print(f"Örnek ses dosyası: {Path(sample_file_path).name}")
    print(f"Örnek ses duygusu: {sample_emotion}")
    print(f"Örnek ses uzunluğu: {duration:.1f} saniye")
    print(f"Örnekleme frekansı: {sample_rate} Hz")

    plt.figure()
    librosa.display.waveshow(signal, sr = sample_rate)
    plt.title(f"Örnek ses dosyası - Duygu: {sample_emotion}")
    plt.xlabel("Zaman (sn)")
    plt.ylabel("Genlik")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close() 

def main() -> None:
    """Veri seti keşif sürecini başlatır"""

    if not RAW_DATA_DIR.exists():
        raise FileNotFoundError("Veri seti klasörü bulunmuyor")
    
    audio_files = find_audio_files(RAW_DATA_DIR) # ravdess klasöründe tüm .wav dosyalarını bulur
    print(f"Bulunan ses dosyası sayısı: {len(audio_files)}")

    if len(audio_files) == 0:
        raise ValueError("Veri klasörü içerisinde .wav dosyası bulunamadı")
    
    metadata_df = create_metadata(audio_files) # ses dosyalarından duygu etiketleri ve dosya bilgilerini içeren metadata oluştur
    print(metadata_df.head())

    print_dataset_summary(metadata_df) # veri setindeki örnek sayısı, sınıf sayısı ve dağılım

    metadata_save_path = PROCESSED_DATA_DIR / "metadata.csv"
    save_metadata(metadata_df, metadata_save_path)

    emotion_distribution_path = PLOTS_DIR / "emotion_distribution.png" # sınıf dağılım grafiği için hedef dosya yolu
    plot_emotiın_distribution(metadata_df, emotion_distribution_path)

    waveform_path = PLOTS_DIR / "sample_waveform.png"
    plot_sample_waveform(metadata_df, waveform_path)

if __name__ == "__main__":
    main()


