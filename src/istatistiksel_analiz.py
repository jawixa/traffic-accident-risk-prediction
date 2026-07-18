# -*- coding: utf-8 -*-
"""
İstatistiksel Analiz + Otomatik Yorumlama
@author: JaWiXa
"""

import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

# Veri yükleme
df = pd.read_csv("trafik_kazasi_verileri.csv")

# === SAYISAL DEĞİŞKENLER - PEARSON KORELASYON ===
sayisal_sutunlar = ['saat', 'sicaklik', 'gorus_mesafesi', 'trafik_yogunlugu',
                    'hiz_limiti', 'ortalama_hiz', 'kavsak_yogunlugu', 'risk_score']

print("=== PEARSON KORELASYON ANALİZİ VE YORUMLARI ===")
for sutun in sayisal_sutunlar:
    corr, pdegeri = stats.pearsonr(df[sutun], df['kaza'])
    
    if pdegeri < 0.05:
        yorum = f"{sutun} değişkeni ile kaza arasında istatistiksel olarak anlamlı bir ilişki vardır."
        if corr > 0:
            yon = "pozitif"
        else:
            yon = "negatif"
        guc = "güçlü" if abs(corr) > 0.5 else "orta" if abs(corr) > 0.3 else "zayıf"
        print(f"- {sutun:20s} | korelasyon: {corr:.3f} | p-değeri: {pdegeri:.5f}")
        print(f"  → {yorum} İlişki yönü {yon}, ilişki gücü {guc} düzeydedir.\n")
    else:
        print(f"- {sutun:20s} | korelasyon: {corr:.3f} | p-değeri: {pdegeri:.5f}")
        print(f"  → {sutun} değişkeni ile kaza arasında anlamlı bir ilişki bulunmamıştır.\n")


# === KATEGORİK DEĞİŞKENLER - ANOVA TESTİ ===
kategorik_sutunlar = ['gun_tipi', 'yagis', 'yol_tipi', 'zemin_durumu', 'isiklandirma', 'is_cikisi_saati']

print("\n=== ANOVA (F-Test) ANALİZİ VE YORUMLARI ===")
for sutun in kategorik_sutunlar:
    gruplar = [df[df[sutun] == deger]['risk_score'] for deger in df[sutun].unique()]
    fstat, pdegeri = stats.f_oneway(*gruplar)
    
    if pdegeri < 0.05:
        print(f"- {sutun:20s} | F-istatistiği: {fstat:.3f} | p-değeri: {pdegeri:.5f}")
        print(f"  → {sutun} değişkeninin grupları arasında risk skorları açısından anlamlı fark vardır.")
        print(f"    Bu, {sutun} değişkeninin kazaları etkileyen önemli bir faktör olduğunu gösterir.\n")
    else:
        print(f"- {sutun:20s} | F-istatistiği: {fstat:.3f} | p-değeri: {pdegeri:.5f}")
        print(f"  → {sutun} değişkeninin grupları arasında anlamlı fark bulunmamıştır.\n")

print("--- YORUM ---")
print("Pearson testinde p < 0.05 → güçlü doğrusal ilişki anlamına gelir.")
print("ANOVA testinde p < 0.05 → değişkenin grupları arasında risk açısından anlamlı fark vardır.")

corr_matrix = df[sayisal_sutunlar + ['kaza']].corr()
plt.figure(figsize=(10,6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
plt.title("Sayısal Değişkenler ve Kaza Arasındaki Korelasyon")
plt.show()
