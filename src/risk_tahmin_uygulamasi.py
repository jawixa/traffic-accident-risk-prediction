# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 19:25:41 2025

@author: JaWiXa
"""

import joblib
import pandas as pd
import numpy as np

# --- 1. MODEL VE VERİ PARAMETRELERİNİN TANIMLANMASI ---
MODEL_DOSYASI = "trafik_kazasi_mlp_model.pkl"
MIN_RISK_ESIGI = 0.55 # Basit bir eşik değeri: Olasılık bu değerin üstündeyse uyarı verilir.

# Kullanıcıdan girdi alırken kolaylık sağlamak için kategorik seçenekler
KATEGORIK_SEVIYELER = {
    'gun_tipi': ['hafta_ici', 'hafta_sonu'],
    'yagis': ['yok', 'hafif', 'siddetli'],
    'yol_tipi': ['sehir_ici', 'otoyol', 'kirsal'],
    'zemin_durumu': ['kuru', 'islak', 'buzlu'],
    'isiklandirma': ['yeterli', 'yetersiz', 'karanlik']
}

# --- 2. UYARI VE ÖNERİ MEKANİZMASI ---

def risk_bazli_oneriler(risk_olasiliği, girdi_verileri):
    """
    Modelin tahmin ettiği risk skoru ve kullanıcı girdilerine göre
    önleyici öneriler listesi oluşturur.
    """
    oneriler = []
    
    # Yüksek Risk Uyarısı
    if risk_olasiliği >= 0.80:
        oneriler.append("🔴 ACİL: Kaza Riski ÇOK YÜKSEK! Lütfen hemen hızınızı düşürün ve en yakın güvenli alanda durun.")
    elif risk_olasiliği >= MIN_RISK_ESIGI:
        oneriler.append("🟠 Yüksek Risk Uyarısı: Dikkatli Olun! Sürüş ve çevre koşulları yüksek kaza riski oluşturuyor.")
    else:
        oneriler.append("🟢 Risk Normal: Sürüş koşulları genel olarak güvenlidir.")
        
    # Koşullara Bağlı Özel Öneriler (Veri setinizdeki faktörlere göre)
    
    # Görüş Mesafesi
    if girdi_verileri['gorus_mesafesi'] < 300:
        if girdi_verileri['yagis'] == 'siddetli' or girdi_verileri['zemin_durumu'] == 'buzlu':
             oneriler.append("❗ Görüş mesafesi düşük. Sis farlarını açın ve takip mesafenizi iki katına çıkarın.")

    # Yağış ve Zemin Durumu
    if girdi_verileri['zemin_durumu'] == 'buzlu' or girdi_verileri['yagis'] == 'siddetli':
        oneriler.append("❄️ Zemin kaygan veya görüş zorlu. Ani fren ve direksiyon hareketlerinden kaçının.")
    elif girdi_verileri['zemin_durumu'] == 'islak':
        oneriler.append("💧 Zemin ıslak. Hızınızı limitin altında tutun ve su birikintilerine dikkat edin (Aquaplaning riski).")

    # Trafik ve Hız
    if girdi_verileri['trafik_yogunlugu'] >= 8 and girdi_verileri['ortalama_hiz'] > girdi_verileri['hiz_limiti'] * 0.9:
         oneriler.append("🚗 Yoğun trafikte yüksek hızdasınız. Lütfen hızınızı azaltın ve şerit değiştirmeyin.")

    # Gece ve Aydınlatma
    if (girdi_verileri['isiklandirma'] in ['yetersiz', 'karanlik']) and girdi_verileri['saat'] >= 18 or girdi_verileri['saat'] <= 6:
        oneriler.append("🌃 Gece sürüşü/karanlıkta görüş mesafesini kontrol edin, uzun far kullanmaktan çekinmeyin.")

    return oneriler

# --- 3. KULLANICI GİRDİSİ ALMA FONKSİYONU ---

def girdi_al_ve_hazirla():
    """Kullanıcıdan gerekli tüm parametreleri alır."""
    print("\n--- PARAMETRE GİRİŞİ ---")
    
    girdi = {}
    
    # Sayısal Girdiler
    girdi['saat'] = int(input("Saat (0-23): "))
    girdi['sicaklik'] = float(input("Sıcaklık (°C): "))
    girdi['gorus_mesafesi'] = int(input("Görüş Mesafesi (metre): "))
    girdi['trafik_yogunlugu'] = int(input("Trafik Yoğunluğu (1: Düşük - 10: Çok Yüksek): "))
    girdi['hiz_limiti'] = int(input("Hız Limiti (km/s): "))
    girdi['ortalama_hiz'] = float(input("Ortalama Sürüş Hızı (km/s): "))
    girdi['kavsak_yogunlugu'] = int(input("Kavşak Yoğunluğu (1: Düşük - 10: Çok Yüksek): "))
    
    # İkili Girdi
    is_cikis = input("İş Çıkışı Saati mi? (E/H): ").strip().lower()
    girdi['is_cikisi_saati'] = 1 if is_cikis == 'e' else 0

    # Kategorik Girdiler (Seçeneklerle)
    for sutun, secenekler in KATEGORIK_SEVIYELER.items():
        print(f"\n{sutun.replace('_', ' ').title()} ({', '.join(secenekler)}):")
        deger = (input("Seçiminizi girin: ")
         .strip()
         .lower()
         .replace("ı","i")
         .replace("ö","o")
         .replace("ü","u"))
        # Girdinin geçerli olup olmadığını basitçe kontrol et
        if deger not in secenekler:
             print(f"Uyarı: Geçersiz değer '{deger}'. Varsayılan olarak '{secenekler[0]}' kullanılıyor.")
             girdi[sutun] = secenekler[0]
        else:
             girdi[sutun] = deger

    return girdi

# --- 4. ANA ÇALIŞTIRMA FONKSİYONU ---

def main():
    """Modeli yükler, tahmin yapar ve sonuçları kullanıcıya sunar."""
    try:
        # Kaydedilmiş modeli yükle (Pre-processor ve MLP dahil)
        model = joblib.load(MODEL_DOSYASI)
        print(f"✅ Model başarıyla yüklendi: {MODEL_DOSYASI}")
    except FileNotFoundError:
        print(f"❌ Hata: Model dosyası bulunamadı: {MODEL_DOSYASI}")
        print("Lütfen MLP modelini eğittiğiniz kodu çalıştırdığınızdan emin olun.")
        return
        
    try:
        yeni_veri_dict = girdi_al_ve_hazirla()
        
        # DataFrame'i oluştur (Modelin beklediği format)
        yeni_veri_df = pd.DataFrame([yeni_veri_dict])
        
        # Tahmin ve Olasılık Hesapla
        kaza_tahmini = model.predict(yeni_veri_df)[0]
        risk_olasiliği = model.predict_proba(yeni_veri_df)[0][1]
        
        # Önerileri Al
        oneriler = risk_bazli_oneriler(risk_olasiliği, yeni_veri_dict)

        # Sonuçları Sun
        print("\n" + "="*50)
        print("          🚨 TRAFİK KAZASI RİSK TAHMİNİ 🚨")
        print("="*50)
        print(f"Kaza Olasılığı (Risk Skoru): %{risk_olasiliği*100:.2f}")
        
        if kaza_tahmini == 1:
             print("Tahmini Durum: ❗ KAZA İHTİMALİ YÜKSEK (Model Tahmini)")
        else:
             print("Tahmini Durum: ✅ Kaza İhtimali DÜŞÜK (Model Tahmini)")
        
        print("\n--- UYARI VE ÖNERİLER ---")
        for i, oneri in enumerate(oneriler):
            print(f"{i+1}. {oneri}")
            
        print("="*50)
        
    except Exception as e:
        print(f"\n❌ Bir hata oluştu: {e}")
        print("Lütfen girdi formatlarınızı kontrol edin.")

if __name__ == "__main__":
    main()