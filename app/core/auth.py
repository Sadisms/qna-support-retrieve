from fastapi import HTTPException, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from app.core.config import get_config


security = HTTPBearer()
config = get_config()


def verify_token(credentials: HTTPAuthorizationCredentials) -> bool:
    if not credentials or credentials.credentials != config.api_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return True


def get_current_user(credentials: HTTPAuthorizationCredentials = security):
    return verify_token(credentials)
