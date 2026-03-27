"""
routes/analytics.py — Usage analytics & statistics
"""
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, case, and_
from datetime import datetime, timedelta, timezone

from database import get_db
from auth.dependencies import get_current_user
from auth.schemas import CurrentUser, UserRole
from models.scan_record import ScanRecord
from auth.models import User

router = APIRouter(prefix="/analytics", tags=["Analytics"])


@router.get("/overview", summary="User's personal analytics overview")
async def user_analytics(
    current_user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    user_q = select(ScanRecord).where(ScanRecord.user_id == current_user.id)
    
    # Total scans
    total = (await db.execute(select(func.count()).select_from(user_q.subquery()))).scalar_one()

    # Tumor detected count
    detected = (await db.execute(
        select(func.count()).where(
            and_(ScanRecord.user_id == current_user.id, ScanRecord.tumor_detected == True)
        )
    )).scalar_one()

    # Type distribution
    type_dist_q = (
        select(ScanRecord.tumor_type, func.count().label("count"))
        .where(and_(ScanRecord.user_id == current_user.id, ScanRecord.tumor_detected == True))
        .group_by(ScanRecord.tumor_type)
    )
    type_dist = {row.tumor_type: row.count for row in (await db.execute(type_dist_q)).all()}

    # Average confidence
    avg_conf = (await db.execute(
        select(func.avg(ScanRecord.confidence)).where(ScanRecord.user_id == current_user.id)
    )).scalar_one() or 0.0

    # Feedback stats
    confirmed = (await db.execute(
        select(func.count()).where(
            and_(ScanRecord.user_id == current_user.id, ScanRecord.doctor_feedback == "confirmed")
        )
    )).scalar_one()

    rejected = (await db.execute(
        select(func.count()).where(
            and_(ScanRecord.user_id == current_user.id, ScanRecord.doctor_feedback == "rejected")
        )
    )).scalar_one()

    # Last 7 days activity
    seven_days_ago = datetime.now(timezone.utc) - timedelta(days=7)
    recent_q = (
        select(
            func.date(ScanRecord.created_at).label("date"),
            func.count().label("count")
        )
        .where(and_(ScanRecord.user_id == current_user.id, ScanRecord.created_at >= seven_days_ago))
        .group_by(func.date(ScanRecord.created_at))
        .order_by("date")
    )
    daily_activity = [
        {"date": str(row.date), "count": row.count}
        for row in (await db.execute(recent_q)).all()
    ]

    return {
        "total_scans": total,
        "tumor_detected": detected,
        "no_tumor": total - detected,
        "detection_rate": round(detected / total, 4) if total else 0.0,
        "tumor_type_distribution": type_dist,
        "average_confidence": round(avg_conf, 4),
        "feedback": {"confirmed": confirmed, "rejected": rejected, "pending": total - confirmed - rejected},
        "daily_activity_7d": daily_activity,
    }


@router.get("/admin/platform", summary="[Admin] Platform-wide statistics")
async def admin_platform_stats(
    current_user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    if current_user.role != UserRole.admin:
        from fastapi import HTTPException
        raise HTTPException(status_code=403, detail="Admin access required.")

    total_users = (await db.execute(select(func.count(User.id)))).scalar_one()
    total_scans = (await db.execute(select(func.count(ScanRecord.id)))).scalar_one()
    total_detected = (await db.execute(
        select(func.count()).where(ScanRecord.tumor_detected == True)
    )).scalar_one()

    # Top 5 active users
    top_users_q = (
        select(User.full_name, User.email, func.count(ScanRecord.id).label("scan_count"))
        .join(ScanRecord, User.id == ScanRecord.user_id)
        .group_by(User.id)
        .order_by(func.count(ScanRecord.id).desc())
        .limit(5)
    )
    top_users = [
        {"full_name": r.full_name, "email": r.email, "scans": r.scan_count}
        for r in (await db.execute(top_users_q)).all()
    ]

    # Type distribution platform-wide
    type_dist_q = (
        select(ScanRecord.tumor_type, func.count().label("count"))
        .where(ScanRecord.tumor_detected == True)
        .group_by(ScanRecord.tumor_type)
    )
    type_dist = {row.tumor_type: row.count for row in (await db.execute(type_dist_q)).all()}

    avg_conf = (await db.execute(select(func.avg(ScanRecord.confidence)))).scalar_one() or 0.0

    return {
        "total_users": total_users,
        "total_scans": total_scans,
        "total_detected": total_detected,
        "detection_rate": round(total_detected / total_scans, 4) if total_scans else 0.0,
        "average_confidence": round(avg_conf, 4),
        "tumor_type_distribution": type_dist,
        "top_active_users": top_users,
    }
