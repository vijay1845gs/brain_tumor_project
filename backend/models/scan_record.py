"""
models/scan_record.py — ORM model for storing scan history
"""
from sqlalchemy import Column, Integer, String, Float, Boolean, Text, DateTime, ForeignKey
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from database import Base


class ScanRecord(Base):
    __tablename__ = "scan_records"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)

    # Prediction results
    tumor_detected = Column(Boolean, nullable=False)
    tumor_type = Column(String(50), nullable=True)   # None if no tumor
    confidence = Column(Float, nullable=False)
    uncertainty = Column(Float, nullable=False)
    reliability = Column(String(120), nullable=False)
    risk_level = Column(String(50), nullable=False)
    risk_color = Column(String(20), nullable=False)
    clinical_note = Column(Text, nullable=False)
    recommendation = Column(Text, nullable=False)
    heatmap_image = Column(Text, nullable=True)      # base64 PNG (large)

    # Feedback from clinician
    doctor_feedback = Column(String(20), nullable=True)   # "confirmed" | "rejected" | None
    feedback_notes = Column(Text, nullable=True)

    # Metadata
    original_filename = Column(String(255), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationship back to user
    user = relationship("User", back_populates="scans")
