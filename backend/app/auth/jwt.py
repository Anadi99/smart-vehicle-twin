"""
JWT authentication — issues and validates Bearer tokens.
Demo credentials: admin / svtwin2026  (set via env vars in production)
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

SECRET_KEY = os.environ.get("JWT_SECRET", "change-me-in-production-svt2026")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.environ.get("JWT_EXPIRE_MINUTES", "60"))

# Demo user store — replace with DB in production
_USERS: dict[str, str] = {
    os.environ.get("ADMIN_USER", "admin"): os.environ.get("ADMIN_PASS", "svtwin2026"),
}

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/token", auto_error=False)


def _try_import_jose():  # type: ignore[return]
    try:
        from jose import JWTError, jwt as _jwt  # type: ignore[import]
        return _jwt, JWTError
    except ImportError:
        return None, None


def create_access_token(subject: str) -> str:
    _jwt, _ = _try_import_jose()
    if _jwt is None:
        return f"demo-token-{subject}"
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    return _jwt.encode({"sub": subject, "exp": expire}, SECRET_KEY, algorithm=ALGORITHM)


def verify_token(token: str) -> str:
    """Return username or raise 401."""
    _jwt, JWTError = _try_import_jose()
    if _jwt is None:
        # jose not installed — accept demo tokens in dev
        if token.startswith("demo-token-"):
            return token.replace("demo-token-", "")
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="python-jose not installed")
    try:
        payload = _jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub", "")
        if not username:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
        return username
    except JWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")


def authenticate_user(username: str, password: str) -> bool:
    return _USERS.get(username) == password


async def get_current_user(token: str | None = Depends(oauth2_scheme)) -> str:
    if token is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return verify_token(token)
