from flask import Flask, render_template, request
import joblib
import pandas as pd
from database import init_db, add_record, get_history
from datetime import datetime
from flask import redirect, url_for, flash
from database import init_db, add_record, get_history, delete_record, clear_all




app = Flask(__name__)

init_db()

# Modeli yükle
model = joblib.load("trafik_kazasi_mlp_model.pkl")

# Ana sayfa
@app.route("/")
def index():
    return render_template("index.html")

# Tahmin
@app.route("/predict", methods=["POST"])
def predict():

    form = request.form

    veri = pd.DataFrame([{
        "saat": int(form['saat_aralik']),
        "sicaklik": float(form['sicaklik']),
        "gorus_mesafesi": int(form['gorus_mesafesi']),
        "trafik_yogunlugu": int(form['trafik_yogunlugu']),
        "hiz_limiti": int(form['hiz_limiti']),
        "ortalama_hiz": float(form['ortalama_hiz']),
        "kavsak_yogunlugu": int(form['kavsak_yogunlugu']),
        "is_cikisi_saati": int(form.get('is_cikisi_saati', 0)),
        "gun_tipi": form['gun_tipi'],
        "yagis": form['yagis'],
        "yol_tipi": form['yol_tipi'],
        "zemin_durumu": form['zemin_durumu'],
        "isiklandirma": form['isiklandirma']
    }])

    tahmin = int(model.predict(veri)[0])
    olasilik = round(float(model.predict_proba(veri)[0][1] * 100), 2)

    # Risk seviyesi
    if olasilik >= 80:
        risk_durumu = "🔴 Çok Yüksek Risk"
        suggestions = ["Hemen hızınızı düşürün ve güvenli bir noktada durun!"]
    elif olasilik >= 55:
        risk_durumu = "🟠 Yüksek Risk"
        suggestions = ["Dikkatli sürün, hız ve takip mesafesine dikkat edin."]
    else:
        risk_durumu = "🟢 Normal Risk"
        suggestions = ["Koşullar genel olarak güvenli."]

    # 🔥 Tahmin geçmişine kaydetme (ARTIK DOĞRU YERDE)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    add_record((
        timestamp,
        olasilik,
        tahmin,
        float(form['sicaklik']),
        float(form['gorus_mesafesi']),
        int(form['trafik_yogunlugu']),
        int(form['hiz_limiti']),
        float(form['ortalama_hiz']),
        int(form['kavsak_yogunlugu']),
        int(form.get('is_cikisi_saati', 0)),
        form['gun_tipi'],
        form['yagis'],
        form['yol_tipi'],
        form['zemin_durumu'],
        form['isiklandirma'],
        int(form['saat_aralik'])
    ))

    return render_template(
        "result.html",
        prediction=tahmin,
        risk_score=olasilik,
        suggestions=suggestions
    )

# Tek bir kaydı silme
@app.route("/delete", methods=["POST"])
def delete():
    from database import delete_record
    record_id = request.form["id"]
    delete_record(record_id)
    return ("", 204)  # Başarılı, sayfa yenileme için JS yönlendirir

# Tüm geçmişi temizle
@app.route("/clear_history", methods=["POST"])
def clear_history():
    from database import clear_all
    clear_all()
    return ("", 204)



@app.route("/history")
def history():
    records = get_history()
    return render_template("history.html", records=records)


if __name__ == "__main__":
    app.run(debug=True)
