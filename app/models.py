from sqlalchemy import Column, Integer, Float, String, DateTime, ForeignKey
from sqlalchemy.sql import func
from .database import Base

class NetworkMetric(Base):
    """Tabel untuk mencatat riwayat metrik telemetri jaringan dari Android"""
    __tablename__ = "network_metrics"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    latency = Column(Float, nullable=False)
    packet_loss = Column(Float, nullable=False)
    download_speed = Column(Float, nullable=False)
    upload_speed = Column(Float, nullable=False)
    connection_status = Column(String, nullable=False)

class AiRecommendation(Base):
    """Tabel untuk mencatat hasil keputusan dan penalaran Agen AI Q-Learning"""
    __tablename__ = "ai_recommendations"

    id = Column(Integer, primary_key=True, index=True)
    metrics_id = Column(Integer, ForeignKey("network_metrics.id"), nullable=False)
    network_state = Column(String, nullable=False)
    recommendation = Column(String, nullable=False)