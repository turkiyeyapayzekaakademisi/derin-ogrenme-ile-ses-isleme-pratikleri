"""
Amaç:
    - Kaggle üzerinden manuel olarak müzik türleri veri setini indir, ses dosyalarının yollarını ve etiketlerini ayarla
    - metadata.csv oluştur

Veri seti: https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification
    - Veri seti 10 farklı müzik türünden oluşur; pop, rock ...

Adımlar:
    1. Gerekli kütüphanelerin içeriye aktarılması
    2. Proje klasör yollarının tanımlanması
    3. Gerekli klasörlerin oluşturulması
    4. Veri seti tür isimlerinin tanımlanması
    5. Tür klasörlerindeki ses dosyalarının okunması
    6. Metadata tablosunun oluşturulması
    7. Metadata dosyasının kaydedilmesi
"""
# 1. Gerekli kütüphanelerin içeriye aktarılması
from pathlib import Path # Dosya ve klasör yollarını işletim sisteminden bağımsız olarak yönetmek için kullanılır
import pandas as pd # dosya yolu ve etiket bilgilerini tablo halinde saklamak için

# 2. Proje klasör yollarının tanımlanması
PROJECT_DIR = Path(__file__).resolve().parent # Bu python dosyasının bulunduğu klasörü proje ana klasörü olarak alır
DATASET_DIR = PROJECT_DIR / "dataset" / "genres_original" # veri seti klasörü
PROCESSED_DIR = PROJECT_DIR / "data" / "processed" # işlenmiş dosya kaydedilecek klasör
METADATA_PATH = PROCESSED_DIR / "metadata.csv" # ses dosyası yolu ve etiket bilgilerinin kaydedileceği csv dosyası

# 3. Gerekli klasörlerin oluşturulması
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# 4. Veri seti tür isimlerinin tanımlanması
genre_names = {
    "blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"
}

# 5. Tür klasörlerindeki ses dosyalarının okunması
metadata_rows = [] # her ses dosyası için dosya yolu ve etiket bilgisi saklamak için liste

for label_id, genre_name in enumerate(genre_names): # her müzik türü için sayısal etiket ve müzik türünü birlikte dolaşalım
    genre_dir = DATASET_DIR / genre_name # ilgili müzik türüne ait klasör yolu
    audio_files = sorted(genre_dir.glob("*.wav")) # ilgili tür klasörlerindeki wav dosyalarını sıralı şekilde alır
    
    for audio_path in audio_files: # ilgili tür klasöründeki her ses dosyası içinde dolaşır
        metadata_rows.append(
            {
                "file_path": str(audio_path), # ses dosyasının tam dosya yolunu metin olarak kaydeder
                "file_name": audio_path.name, # ses dosyasının sadece dosya adını kaydeder
                "label": label_id, # müzik türünün sayısal etiket
                "label_name": genre_name # müzik türünün metinsel sınıf adı 
            }
        )

# 6. Metadata tablosunun oluşturulması
metadata_df = pd.DataFrame(metadata_rows) # toplanan dosya yolu ve etiket bilgilerini pandas dataframe formatına dönüştür
print(metadata_df.head())

# 7. Metadata dosyasının kaydedilmesi
metadata_df.to_csv(METADATA_PATH, index=False, encoding="utf-8")

print(f"metadata_df length: {len(metadata_df)}")