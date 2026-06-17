"""
Amaç:
    - Eğitilen autoencoder modelini test verisi üzerinden değerlendir.
    - Gürültü ses, temizlenmiş ve gerçek temiz ses arasındaki farkı MSE, MAE, SNR, SI-SDR ve SI-SDRi metrikleri ile karşılaştır

Veri seti:
    - gürültülü veri seti, temizlenmiş veri seti, gerçek temiz veri seti

Adımlar:
    1. Gerekli kütüphanelerin içeriye aktarılması
    2. Proje klasör yollarının ve değerlendirme ayarlarının tanımlanması
    3. Gerekli klasörlerin oluşturulması
    4. Test verilerinin ve eğitilmiş modelin yüklenmesi
    5. Değerlendirme metriklerinin tanımlanması
    6. Gürültülü verileri kullanarak model ile temizlenmiş verilerin üretilmesi
    7. Tüm ses örnekleri için metriklerin hesaplanması
    8. Ortalama değerlendirme sonuçlarının ekrana yazdırılması
    9. Örnek ses dosyalarının kaydedilmesi
    10. Değerlendirme sonuçlarının csv dosyasına kaydedilmesi
    11. Ana çalışma akışının oluşturulması
"""
# 1. Gerekli kütüphanelerin içeriye aktarılması
from pathlib import Path
import csv
import numpy as np
import soundfile as sf
from tqdm import tqdm
from tensorflow.keras.models import load_model # eğitilmiş keras modelini yüklemek için kullanılır

# 2. Proje klasör yollarının ve değerlendirme ayarlarının tanımlanması
BASE_DIR = Path(__file__).resolve().parent # Bu python dosyasının bulunduğu klasörü proje ana klasörü olarak alır
DATA_DIR = BASE_DIR / "data" # tüm veri dosyaları için ana klasör
PROCESSED_DATA_DIR = DATA_DIR / "processed" # model eğitime hazır numpy dosyalarının kaydedileceği klasör

MODELS_DIR = BASE_DIR / "models" # eğitilen model dosyalarının kaydedileceği klasör
OUTPUTS_DIR = BASE_DIR / "outputs" # Eğitim çıktılarımızın kaydedileceği ana klasör
AUDIO_EXAMPLES_DIR = OUTPUTS_DIR / "audio_examples" # örnek ses dosyalarının kaydedileceği klasör

MODELS_PATH = MODELS_DIR / "audio_denoising_autoencoder.keras" # eğitilen modelin kaydedileceği dosya yolu
RESULTS_CSV_PATH = OUTPUTS_DIR / "evaluation_results.csv" # değerlendirme sonuçlarının kaydedileceği csv

SAMPLE_RATE = 16000 # Hz ses dosyası frekansı
BATCH_SIZE = 32 
EXAMPLE_COUNT = 5 # kaydedilecek örnek ses kayıtları sayısı (temiz, gürültülü, temizlenmiş)
EPSILON = 1e-8 # bölme işlemlerinde sıfıra bölme hatası almamak için kullandığımız küçük sabit değer

# 3. Gerekli klasörlerin oluşturulması
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
AUDIO_EXAMPLES_DIR.mkdir(parents=True, exist_ok=True) 

# 4. Test verilerinin ve eğitilmiş modelin yüklenmesi
def load_test_data_and_model():
    """Modeli ve test verisini yükler"""

    X_test = np.load(PROCESSED_DATA_DIR / "X_test.npy")
    y_test = np.load(PROCESSED_DATA_DIR / "y_test.npy")
    model = load_model(MODELS_PATH)

    return X_test, y_test, model

# 5. Değerlendirme metriklerinin tanımlanması
def prepare_audio(audio): 
    """Ses verisini metrik hesaplamaya uygun tek boyutlu hale getiren fonksiyonu tanımlar"""
    audio = np.squeeze(audio) # ses verisindeki gereksiz kanal boyutunu kaldır 
    audio = audio.astype(np.float32) # ses verisini float32 formatına dönüştür
    return audio

def calculate_mse(clean_audio, compared_audio):
    """Temiz ses ile karşılaştırılan ses arasındaki mse değerini hesapla"""
    mse_value = np.mean((clean_audio - compared_audio)**2) # ortalama karesel hatayı hesapla
    return mse_value

def calculate_mae(clean_audio, compared_audio): 
    """Temiz ses ile karşılaştırılan ses arasındaki mae değerini hesapla"""
    mae_value = np.mean(np.abs(clean_audio - compared_audio)) # ortalama mutlak hata
    return mae_value

def calculate_snr(clean_audio, compared_audio): 
    """Temiz ses ile karşılaştırılan ses arasındaki snr değerini hesapla"""
    signal_power = np.sum(clean_audio**2) # temiz sesin toplam sinyal gücü
    noise_power = np.sum((clean_audio - compared_audio) ** 2) # temiz ses ile karşılaştırılan ses arasındaki hata gücü
    snr_value = 10 * np.log10((signal_power + EPSILON)/(noise_power + EPSILON)) # SNR ı dB cinsinden hesaplar
    return snr_value

def calculate_si_sdr(clean_audio, compared_audio):
    """Temiz ses ile karşılaştırılan ses arasındaki SI-SDR değerini hesapla"""
    clean_audio = clean_audio - np.mean(clean_audio) # temiz sesi sıfır ortalamalı hale getir
    compared_audio = compared_audio - np.mean(compared_audio) # karşılaştırılan sesi sıfır ortalamalı hale getir

    clean_energy = np.sum(clean_audio**2) + EPSILON # temiz sesin enerjisini hesaplar
    scale = np.sum(compared_audio * clean_audio) / clean_energy # karşılaştırılan sesi temiz ses ölçeklemek için katsayı hesabı
    target_audio = scale * clean_audio # temiz ses doğrultusunda ki hedef bileşen
    noise_audio = compared_audio - target_audio # hedef bileşen dışında kalan hata bileşenini hesaplar

    target_power = np.sum(target_audio ** 2) # hedef bileşenin gücü
    noise_power = np.sum(noise_audio ** 2) # hata bileşenin gücü
    si_sdr_value = 10 * np.log10((target_power + EPSILON) / (noise_power + EPSILON)) # SI-SDR değerini db cinsinden hesaplar

    return si_sdr_value

# 6. Gürültülü verileri kullanarak model ile temizlenmiş verilerin üretilmesi
def predict_denoised_audio(model, X_test):
    """Eğitilmiş model ile temizlenmiş sesleri üret"""
    denoised_audio = model.predict(X_test, batch_size = BATCH_SIZE, verbose = 1) # gürültülü seslerden temizlenmiş sesleri üret
    denoised_audio = np.clip(denoised_audio, -1, 1) # model çıktısını geçerli ses genlik aralığına sıkıştır
    print(f"Temizleniş seslerin şekli: {denoised_audio.shape}")
    return denoised_audio

# 7. Tüm ses örnekleri için metriklerin hesaplanması
def calculate_all_metrics(X_test, y_test, denoised_audio):
    """Tüm test örnekleri için değerlendirme metriklerini hesaplar"""

    results = [] # test sonuçlarını saklar

    for index in tqdm(range(len(X_test))):

        noisy = prepare_audio(X_test[index]) # kirli ses örneğini tek boyutlu hale getir
        clean = prepare_audio(y_test[index]) # gerçek temiz ses örneğini tek boyutlu hale getir
        denoised = prepare_audio(denoised_audio[index]) # modelin ürettiği temizlenmiş sesi tek boyutlu hale getirir

        noisy_mse = calculate_mse(clean, noisy) # kirli ses ile temiz ses arasında mse
        denoised_mse = calculate_mse(clean, denoised) # temizlenmiş ses ile temiz ses arasında mse

        noisy_mae = calculate_mae(clean, noisy) # kirli ses ile temiz ses arasında mae
        denoised_mae = calculate_mae(clean, denoised) # temizlenmiş ses ile temiz ses arasında mae

        noisy_snr = calculate_snr(clean, noisy) # kirli ses ile temiz ses arasında snr
        denoised_snr = calculate_snr(clean, denoised) # temizlenmiş ses ile temiz ses arasında snr
        snr_improved = denoised_snr - noisy_snr # temizlenmiş sesin kirli sese göre snr iyileştirmesini hesaplar

        noisy_si_sdr = calculate_si_sdr(clean, noisy) # kirli ses ile temiz ses arasında si_sdr
        denoised_si_sdr = calculate_si_sdr(clean, denoised) # temizlenmiş ses ile temiz ses arasında si_sdr
        si_srd_improved = denoised_si_sdr - noisy_si_sdr # temizlenmiş sesin kirli sese göre si-srd iyileştirmesi iyileştirmesini hesaplar

        result = {
            "sample_index": index, # test örneğinin indekslerini hesaplar
            "noisy_mse": noisy_mse, # kirli sesin mse değeri
            "denoised_mse": denoised_mse, # temizlenmiş sesin mse değeri 
            "noisy_mae": noisy_mae, # kirli sesin mae değeri
            "denoised_mae": denoised_mae, # temizlenmiş sesin mae değeri 
            "noisy_snr": noisy_snr, # kirli sesin snr değeri
            "denoised_snr": denoised_snr, # temizlenmiş sesin snr değeri 
            "snr_improved": snr_improved, # snr iyileştirmesi
            "noisy_si_sdr": noisy_si_sdr, # kirli sesin si_sdr değeri
            "denoised_si_sdr": denoised_si_sdr, # temizlenmiş sesin si_sdr değeri 
            "si_srd_improved": si_srd_improved, # si_sdr iyileştirmesi
        }

        results.append(result)

    return results

# 8. Ortalama değerlendirme sonuçlarının ekrana yazdırılması
def show_average_results(results): 
    """Ortalama değerlendirme sonuçlarını ekrana yazdıran fonksiyonu tanımlar"""

    metric_names = list(results[0].keys()) # sonuç sözlüğünde ki metrik isimlerini liste olarak alır
    metric_names.remove("sample_index") # sample indexi çıkart çünkü metric değil

    for metric_name in metric_names: # her metrik adı üzerinde döngü başlatılır
        metric_values = [result[metric_name] for result in results] # ilgili metriğin tüm test örneklerindeki değerini alır
        metric_mean = np.mean(metric_values)
        print(f"{metric_name}: {metric_mean}") # ortalama metrik değerlerini ekrana yazdır

# 9. Örnek ses dosyalarının kaydedilmesi
def save_audio_example(X_test, y_test, denoised_audio):
    """Dinlemek için örnek temiz, kirli ve temizlenmiş sesleri kaydet"""

    example_count = min(EXAMPLE_COUNT, len(X_test)) # kaydedilecek örnek sayısını test verisi sayısı ile sınırla

    for index in range(example_count):
        noisy = prepare_audio(X_test[index]) # gürültülü ses örneğini tek boyutlu hale getirir
        clean = prepare_audio(y_test[index]) # temiz ses örneğini tek boyutlu hale getirir
        denoised = prepare_audio(denoised_audio[index]) # temizlenmiş ses örneğini tek boyutlu hale getirir

        sf.write(AUDIO_EXAMPLES_DIR / f"sample_{index}_clean.wav", clean, SAMPLE_RATE)
        sf.write(AUDIO_EXAMPLES_DIR / f"sample_{index}_noisy.wav", noisy, SAMPLE_RATE)
        sf.write(AUDIO_EXAMPLES_DIR / f"sample_{index}_denoised.wav", denoised, SAMPLE_RATE)

# 10. Değerlendirme sonuçlarının csv dosyasına kaydedilmesi
def save_results_to_csv(results): 
    """Tüm değerlendirme sonuçlarını csv dosyasına kaydeden fonksiyonumuz"""

    field_names = list(results[0].keys()) # csv dosyası için tüm sütun isimlerini sonuç sözlüğünden al

    with open(RESULTS_CSV_PATH, mode="w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=field_names)
        writer.writeheader()
        writer.writerows(results)

# 11. Ana çalışma akışının oluşturulması
def main():

    X_test, y_test, model = load_test_data_and_model()
    denoised_audio = predict_denoised_audio(model, X_test)
    results = calculate_all_metrics(X_test, y_test, denoised_audio)
    show_average_results(results)
    save_audio_example(X_test, y_test, denoised_audio)
    save_results_to_csv(results)

if __name__ == "__main__":
    main()

"""
noisy_mse: 0.001211080583743751
denoised_mse: 0.0005760128260590136


noisy_mae: 0.026281556114554405
denoised_mae: 0.012628063559532166


noisy_snr: 9.54851245880127
denoised_snr: 13.971983909606934
snr_improved: 4.423470497131348


noisy_si_sdr: 9.577492713928223
denoised_si_sdr: 13.599084854125977
si_srd_improved: 4.02159309387207

"""