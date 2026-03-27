"""
routes/history.py — Scan history management
"""
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, desc
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime

from database import get_db
from auth.dependencies import get_current_user
from auth.schemas import CurrentUser
from models.scan_record import ScanRecord

router = APIRouter(prefix="/history", tags=["History"])


# ── Schemas ──────────────────────────────────────────────────────────────────

class ScanSummary(BaseModel):
    id: int
    tumor_detected: bool
    tumor_type: Optional[str]
    confidence: float
    uncertainty: float
    risk_level: str
    risk_color: str
    reliability: str
    doctor_feedback: Optional[str]
    original_filename: Optional[str]
    created_at: datetime

    model_config = {"from_attributes": True}


class ScanDetail(ScanSummary):
    clinical_note: str
    recommendation: str
    heatmap_image: Optional[str]
    feedback_notes: Optional[str]


class FeedbackRequest(BaseModel):
    feedback: str          # "confirmed" | "rejected"
    notes: Optional[str] = None


class PaginatedScans(BaseModel):
    total: int
    page: int
    page_size: int
    items: List[ScanSummary]


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("", response_model=PaginatedScans, summary="List scan history")
async def list_history(
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=50),
    tumor_type: Optional[str] = Query(None),
    current_user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    offset = (page - 1) * page_size
    q = select(ScanRecord).where(ScanRecord.user_id == current_user.id)
    if tumor_type:
        q = q.where(ScanRecord.tumor_type == tumor_type)
    
    total_q = select(func.count()).select_from(q.subquery())
    total = (await db.execute(total_q)).scalar_one()
    
    items_q = q.order_by(desc(ScanRecord.created_at)).offset(offset).limit(page_size)
    items = (await db.execute(items_q)).scalars().all()
    
    return PaginatedScans(total=total, page=page, page_size=page_size, items=items)


@router.get("/{scan_id}", response_model=ScanDetail, summary="Get scan detail")
async def get_scan(
    scan_id: int,
    current_user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    scan = await db.get(ScanRecord, scan_id)
    if not scan or scan.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Scan record not found.")
    return scan


@router.delete("/{scan_id}", status_code=204, summary="Delete a scan record")
async def delete_scan(
    scan_id: int,
    current_user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    scan = await db.get(ScanRecord, scan_id)
    if not scan or scan.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Scan record not found.")
    await db.delete(scan)
    await db.commit()


@router.post("/{scan_id}/feedback", summary="Submit doctor feedback on a scan")
async def submit_feedback(
    scan_id: int,
    body: FeedbackRequest,
    current_user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    if body.feedback not in ("confirmed", "rejected"):
        raise HTTPException(status_code=400, detail="feedback must be 'confirmed' or 'rejected'.")
    
    scan = await db.get(ScanRecord, scan_id)
    if not scan or scan.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Scan record not found.")
    
    scan.doctor_feedback = body.feedback
    scan.feedback_notes = body.notes
    await db.commit()
    return {"message": "Feedback recorded.", "scan_id": scan_id}
