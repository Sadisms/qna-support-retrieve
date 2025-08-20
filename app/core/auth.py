from typing import Optional
from fastapi import HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from app.core.config import get_config


security = HTTPBearer()
config = get_config()


def get_workspace_from_token(token: str) -> Optional[str]:
    if token == config.api_token:
        return None
    
    for workspace_id, workspace_token in config.workspace_tokens.items():
        if token == workspace_token:
            return workspace_id
    
    return None


def verify_token(credentials: HTTPAuthorizationCredentials) -> str:
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    workspace_id = get_workspace_from_token(credentials.credentials)
    if workspace_id is None and credentials.credentials != config.api_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return workspace_id


def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Optional[str]:
    return verify_token(credentials)


def get_current_workspace(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    workspace_id = verify_token(credentials)
    if workspace_id is None:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin token cannot be used for workspace-specific operations. Use workspace token."
        )
    return workspace_id
