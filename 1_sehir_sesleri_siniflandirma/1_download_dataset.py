"""
Amaç:
    ESC-50 veri setini indirip, zip dosyasını aç ve proje klasörü içerisinde kullanıma hazır hale getir.

Veri seti:
    1. ESC-50, çevresel ses sınıflandırma için kullanılan bir veri seti
    2. 2000 adet wav formatında ses dosyası içerir.
    3. Her ses yaklaşık 5 saniye
    4. Veri setinde 50 farklı sınıf bulunuyor ve her sınıftan 40 örnek var.

Adımlar:
    1. Gerekli kütüphanelerin içeriye aktarılması
    2. Veri seti bağlantısının ve klasör yollarının tanımlanması
    3. Dosya indirme fonksiyonunu hazırlanması
    4. Zip dosyasını çıkarma fonksiyonun hazırlanması
    5. Ana çalışma akışının oluşturulması
    6. Veri setinin indirilmesi
    7. Zip dosyasının çıkarılması
    8. Sonuç yollarının ekrana yazdırılması    
"""

# 1. Gerekli kütüphanelerin içeriye aktarılması
import os # dosya ve klasör işleri için 
import zipfile # zip dosyalarını açmak için
import requests # internetten dosya indirmek sorgu atmak
from tqdm import tqdm # ilerleme çubuğu

# 2. Veri seti bağlantısının ve klasör yollarının tanımlanması
DATASET_URL = "https://github.com/karolpiczak/ESC-50/archive/refs/heads/master.zip" # esc-50 veri seti indirme bağlantısı
RAW_DATA_DIR = "data/raw" # ham veri dosyası
ZIP_DATA = os.path.join(RAW_DATA_DIR, "esc50.zip") # indirilecek zip dosyası kayıt yolu

# 3. Dosya indirme fonksiyonunu hazırlanması
def download_file(url, save_path):
    """
        Verilen URL'de ki dosyayı indirir ve belirtilen yola kaydeder
    """

    response = requests.get(url, stream=True) # dosyası parça parça indirmek için istek gönder
    response.raise_for_status() # indirme isteğinde hata varsa progromı durdurur
    total_size = int(response.headers.get("content-length",0)) # dosyasının toplam boyutunun byte cinsinden alınması
    with open(save_path, "wb") as file: 
        with tqdm(
            total=total_size,
            unit="B",
            unit_scale=True,
            desc="Veri seti indiriliyor"
        ) as progress_bar:
            for chunk in response.iter_content(chunk_size=1024): # dosyayı 1024 bytelik parçalar halinde indirir
                if chunk:
                    file.write(chunk) # indirilen parçayı dosyaya yazar
                    progress_bar.update(len(chunk)) 

# 4. Zip dosyasını çıkarma fonksiyonun hazırlanması
def extract_zip(zip_path, extract_to):
    """
    Zip dosyasını belirtilen klasöre çıkart
    """

    with zipfile.ZipFile(zip_path, "r") as zip_ref: # zip dosyasını okuma modunda aç
        zip_ref.extractall(extract_to) # zip dosyasında ki tüm içeriği hedef klasöre çıkart

# 5. Ana çalışma akışının oluşturulması
def main():
    """
    Veri setini indirme ve çıkarma işlemlerini sırasıyla çalıştırır
    """
    # 6. Veri setinin indirilmesi
    os.makedirs(RAW_DATA_DIR, exist_ok=True) # data/raw klasörünü yoksa oluştur

    if not os.path.exists(ZIP_DATA): # zip dosyası daha önce indirilmemişse
        print("ESC-50 veri seti indiriliyor")
        download_file(DATASET_URL, ZIP_DATA)
    else:
        print("Zip dosyası zaten mevcut")

    # 7. Zip dosyasının çıkarılması
    extracted_folder = os.path.join(RAW_DATA_DIR, "ESC-50-master")
    if not os.path.exists(extracted_folder): 
        print("Zip dosyası açılıyor")
        extract_zip(ZIP_DATA, RAW_DATA_DIR)
    else:
        print("Veri zaten çıkarılmış")
    
    # 8. Sonuç yollarının ekrana yazdırılması
    audio_path = os.path.join(extracted_folder, "audio") # ses dosyalarının bulunduğu klasör yolu
    metadata_path = os.path.join(extracted_folder, "meta", "esc50.csv")
    print("işlem tamamlandı") 
    print(f"Ses dosyalarının yolu: {audio_path}")   
    print(f"Metadata dosyalarının yolu: {metadata_path}")   

if __name__ == "__main__":
    main()
