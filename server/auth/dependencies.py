"""
Authentication dependencies for protecting routes.
"""

from fastapi import Depends, HTTPException, status, Cookie, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from typing import Optional

from auth.database import get_db
from auth.models import User
from auth.auth_handler import verify_token

security = HTTPBearer()

def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
):
    """Get current authenticated user"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    # Verify token
    token_data = verify_token(credentials.credentials, credentials_exception)
    
    # Get user from database
    user = db.query(User).filter(User.username == token_data["username"]).first()
    if user is None:
        raise credentials_exception
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Inactive user"
        )
    
    return user


def get_current_active_user(current_user: User = Depends(get_current_user)):
    """Get current active user"""
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="Inactive user"
        )
    return current_user


def get_optional_current_user(
    request: Request,
    access_token: Optional[str] = Cookie(None),
    db: Session = Depends(get_db)
):
    """Get current user if token is provided (optional authentication)"""
    token = None
    
    # First try to get token from Authorization header
    auth_header = request.headers.get("authorization")
    if auth_header and auth_header.startswith("Bearer "):
        token = auth_header.replace("Bearer ", "")
    
    # If no header token, try cookie
    elif access_token:
        token = access_token.replace("Bearer ", "") if access_token.startswith("Bearer ") else access_token
    
    # If no token found, return None
    if not token:
        return None
    
    try:
        credentials_exception = HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials"
        )
        
        token_data = verify_token(token, credentials_exception)
        user = db.query(User).filter(User.username == token_data["username"]).first()
        
        if user and user.is_active:
            return user
    except:
        pass
    
    return None 