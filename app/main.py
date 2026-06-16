from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from .database import engine, Base, get_db
from . import models, q_learning
from pydantic import BaseModel

# Inisialisasi pembuatan tabel database SQLite secara otomatis jika belum ada
Base.metadata.create_all(bind=engine)

app = FastAPI(title="AI Mobile Network Monitoring System API")
agent = q_learning.QLearningNetworkAgent()

# Skema Pydantic untuk validasi data request JSON yang masuk dari Android
class MetricCreate(BaseModel):
    latency: float
    packet_loss: float
    download_speed: float
    upload_speed: float
    connection_status: str

@app.post("/metrics")
def create_metrics(metric: MetricCreate, db: Session = Depends(get_db)):
    # 1. Simpan metrik mentah dari Android ke database
    db_metric = models.NetworkMetric(
        latency=metric.latency,
        packet_loss=metric.packet_loss,
        download_speed=metric.download_speed,
        upload_speed=metric.upload_speed,
        connection_status=metric.connection_status
    )
    db.add(db_metric)
    db.commit()
    db.refresh(db_metric)

    # 2. Masukkan data ke modul AI Q-Learning untuk penalaran & pembaruan nilai Q
    state_label, ai_rec = agent.get_recommendation_and_learn(metric.latency, metric.packet_loss)

    # 3. Simpan hasil rekomendasi AI ke database
    db_rec = models.AiRecommendation(
        metrics_id=db_metric.id,
        network_state=state_label,
        recommendation=ai_rec
    )
    db.add(db_rec)
    db.commit()

    # 4. Evaluasi Alert System / Sistem Peringatan Threshold Batas Atas
    is_alert = metric.latency > 100 or metric.packet_loss > 2
    alert_message = "PERINGATAN: Kinerja jaringan menurun terdeteksi!" if is_alert else "Kondisi Aman"

    return {
        "status": "success",
        "metric_id": db_metric.id,
        "network_state": state_label,
        "ai_recommendation": ai_rec,
        "alert": {
            "is_triggered": is_alert,
            "message": alert_message
        }
    }