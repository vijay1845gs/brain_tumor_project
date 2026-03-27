"""
routes/users.py — User profile management & admin user listing
"""
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, desc
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime

from database import get_db
from auth.dependencies import get_current_user
from auth.schemas import CurrentUser, UserRole
from auth.models import User
from auth.service import hash_password, verify_password

router = APIRouter(prefix="/users", tags=["Users"])


# ── Schemas ──────────────────────────────────────────────────────────────────

class ProfileUpdate(BaseModel):
    full_name: Optional[str] = None
    bio: Optional[str] = None


class PasswordChange(BaseModel):
    current_password: str
    new_password: str


class UserOut(BaseModel):
    id: int
    full_name: str
    email: str
    role: UserRole
    bio: Optional[str]
    created_at: datetime

    model_config = {"from_attributes": True}


class PaginatedUsers(BaseModel):
    total: int
    items: List[UserOut]


# ── Own profile ───────────────────────────────────────────────────────────────

@router.get("/me", response_model=UserOut, summary="Get own profile")
async def get_profile(
    current_user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    user = await db.get(User, current_user.id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")
    return user


@router.patch("/me", response_model=UserOut, summary="Update own profile")
async def update_profile(
    body: ProfileUpdate,
    current_user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    user = await db.get(User, current_user.id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")
    if body.full_name is not None:
        if not body.full_name.strip():
            raise HTTPException(status_code=400, detail="Full name cannot be empty.")
        user.full_name = body.full_name.strip()
    if body.bio is not None:
        user.bio = body.bio
    await db.commit()
    await db.refresh(user)
    return user


@router.post("/me/change-password", summary="Change own password")
async def change_password(
    body: PasswordChange,
    current_user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    user = await db.get(User, current_user.id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")
    if not verify_password(body.current_password, user.hashed_password):
        raise HTTPException(status_code=400, detail="Current password is incorrect.")
    if len(body.new_password) < 8:
        raise HTTPException(status_code=400, detail="New password must be at least 8 characters.")
    user.hashed_password = hash_password(body.new_password)
    await db.commit()
    return {"message": "Password updated successfully."}


# ── Admin: list all users ─────────────────────────────────────────────────────

@router.get("", response_model=PaginatedUsers, summary="[Admin] List all users")
async def list_users(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    search: Optional[str] = Query(None),
    current_user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    if current_user.role != UserRole.admin:
        raise HTTPException(status_code=403, detail="Admin access required.")
    offset = (page - 1) * page_size
    q = select(User)
    if search:
        like = f"%{search}%"
        q = q.where(
            User.full_name.ilike(like) | User.email.ilike(like)
        )
    total = (await db.execute(select(func.count()).select_from(q.subquery()))).scalar_one()
    items = (await db.execute(q.order_by(desc(User.created_at)).offset(offset).limit(page_size))).scalars().all()
    return PaginatedUsers(total=total, items=items)


@router.delete("/{user_id}", status_code=204, summary="[Admin] Delete a user")
async def delete_user(
    user_id: int,
    current_user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    if current_user.role != UserRole.admin:
        raise HTTPException(status_code=403, detail="Admin access required.")
    if user_id == current_user.id:
        raise HTTPException(status_code=400, detail="Cannot delete your own account.")
    user = await db.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")
    await db.delete(user)
    await db.commit()


@router.patch("/{user_id}/role", summary="[Admin] Change a user's role")
async def change_user_role(
    user_id: int,
    new_role: UserRole,
    current_user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    if current_user.role != UserRole.admin:
        raise HTTPException(status_code=403, detail="Admin access required.")
    user = await db.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")
    user.role = new_role
    await db.commit()
    return {"message": f"Role updated to {new_role.value}", "user_id": user_id}
