# -*- coding: utf-8 -*-
"""
Yapay Sinir Ağı (MLP) modeli ile trafik kazası tahmini
@author: JaWiXa
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# 1 Veri yükleme
df = pd.read_csv("trafik_kazasi_verileri.csv")

# 2 Bağımsız değişkenler (X) ve bağımlı değişken (y)
X = df.drop(columns=['kaza', 'risk_score', 'suggestions'])  # risk_score ve suggestions eğitimde yok
y = df['kaza']

# 3 Kategorik ve sayısal sütunların ayrımı
kategorik_sutunlar = ['gun_tipi', 'yagis', 'yol_tipi', 'zemin_durumu', 'isiklandirma']
sayisal_sutunlar = ['saat', 'sicaklik', 'gorus_mesafesi', 'trafik_yogunlugu',
            'hiz_limiti', 'ortalama_hiz', 'kavsak_yogunlugu', 'is_cikisi_saati']

# 4 One-Hot Encoding + Standard Scaler
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), sayisal_sutunlar),
    ('cat', OneHotEncoder(handle_unknown='ignore'), kategorik_sutunlar)
])

# 5 Veri bölme (%80 eğitim - %20 test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 6 MLP modeli tanımlama
mlp = MLPClassifier(
    hidden_layer_sizes=(128, 64, 32),  # iki gizli katman
    activation='relu',
    solver='adam',
    max_iter=500,
    random_state=42
)


# 7 Pipeline ile model oluşturma
model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('classifier', mlp)])


# 8 Modeli eğit
print("Model eğitiliyor...")
model.fit(X_train, y_train)

# 9 Test tahminleri
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

if not model.named_steps['classifier'].n_iter_ < mlp.max_iter:
    print("\n⚠️ Uyarı: Model tam yakınsamamış olabilir (max_iter'e ulaştı). max_iter değerini 500–800 aralığına çıkarabilirsin.")


# 🔟 Performans raporu
print("\n=== SINIFLANDIRMA RAPORU ===")
print(classification_report(y_test, y_pred))

print("\n=== KARIŞIKLIK MATRİSİ ===")
print(confusion_matrix(y_test, y_pred))

roc_auc = roc_auc_score(y_test, y_prob)
print(f"\nROC-AUC Skoru: {roc_auc:.3f}")

# 🔁 İleri aşama: modeli kaydedip CLI'da kullanılabilir hale getireceğiz
import joblib
joblib.dump(model, "trafik_kazasi_mlp_model.pkl")
print("\nModel başarıyla kaydedildi: trafik_kazasi_mlp_model.pkl")


