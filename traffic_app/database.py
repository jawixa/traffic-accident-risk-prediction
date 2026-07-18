import sqlite3

def init_db():
    conn = sqlite3.connect("history.db")
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            risk_score REAL,
            prediction INTEGER,
            sicaklik REAL,
            gorus_mesafesi REAL,
            trafik_yogunlugu INTEGER,
            hiz_limiti INTEGER,
            ortalama_hiz REAL,
            kavsak_yogunlugu INTEGER,
            is_cikisi_saati INTEGER,
            gun_tipi TEXT,
            yagis TEXT,
            yol_tipi TEXT,
            zemin_durumu TEXT,
            isiklandirma TEXT,
            saat_aralik INTEGER
        )
    """)

    conn.commit()
    conn.close()


# Veri ekleme
def add_record(data):
    conn = sqlite3.connect("history.db")
    c = conn.cursor()

    c.execute("""
        INSERT INTO predictions (
            timestamp, risk_score, prediction, sicaklik, gorus_mesafesi,
            trafik_yogunlugu, hiz_limiti, ortalama_hiz, kavsak_yogunlugu,
            is_cikisi_saati, gun_tipi, yagis, yol_tipi, zemin_durumu,
            isiklandirma, saat_aralik
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, data)

    conn.commit()
    conn.close()


# Geçmiş kayıtları çekme
def get_history():
    conn = sqlite3.connect("history.db")
    c = conn.cursor()

    c.execute("SELECT * FROM predictions ORDER BY id DESC")
    rows = c.fetchall()

    conn.close()
    return rows


# Tek kayıt silme (DÜZELTİLDİ)
def delete_record(record_id):
    conn = sqlite3.connect("history.db")
    cursor = conn.cursor()
    cursor.execute("DELETE FROM predictions WHERE id = ?", (record_id,))
    conn.commit()
    conn.close()


# Tüm kayıtları silme (DÜZELTİLDİ)
def clear_all():
    conn = sqlite3.connect("history.db")
    cursor = conn.cursor()
    cursor.execute("DELETE FROM predictions")
    conn.commit()
    conn.close()
