from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm

from app.auth.jwt import authenticate_user, create_access_token

router = APIRouter(prefix="/api/auth", tags=["auth"])


@router.post("/token")
def login(form: OAuth2PasswordRequestForm = Depends()) -> dict:
    if not authenticate_user(form.username, form.password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = create_access_token(subject=form.username)
    return {"access_token": token, "token_type": "bearer"}
