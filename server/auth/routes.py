"""
Authentication routes for user registration, login, and management.
"""

from fastapi import APIRouter, Depends, HTTPException, status, Response, Query
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from datetime import timedelta
import secrets
import string

from auth.database import get_db
from auth.models import User
from auth.schemas import UserCreate, UserLogin, UserResponse, Token, OAuthAuthURL, OAuthCallback
from auth.auth_handler import (
    get_password_hash, 
    verify_password, 
    create_access_token,
    ACCESS_TOKEN_EXPIRE_MINUTES
)
from auth.dependencies import get_current_active_user
from auth.oauth_handler import oauth_handler

router = APIRouter(prefix="/auth", tags=["authentication"])


@router.post("/register", response_model=Token)
async def register_user(user: UserCreate, db: Session = Depends(get_db)):
    """Register a new user"""
    try:
        # Hash the password
        hashed_password = get_password_hash(user.password)
        
        # Create new user
        db_user = User(
            username=user.username,
            email=user.email,
            password_hash=hashed_password
        )
        
        db.add(db_user)
        db.commit()
        db.refresh(db_user)
        
        # Create access token
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": db_user.username}, 
            expires_delta=access_token_expires
        )
        
        # Return user data and token
        user_response = UserResponse(
            id=db_user.id,
            username=db_user.username,
            email=db_user.email,
            created_at=db_user.created_at,
            is_active=db_user.is_active,
            oauth_provider=db_user.oauth_provider,
            avatar_url=db_user.avatar_url
        )
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "user": user_response
        }
        
    except IntegrityError:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username or email already registered"
        )


@router.post("/login", response_model=Token)
async def login_user(user_credentials: UserLogin, response: Response, db: Session = Depends(get_db)):
    """Login user and return access token"""
    # Find user by username or email
    db_user = db.query(User).filter(
        (User.username == user_credentials.username) | 
        (User.email == user_credentials.username)
    ).first()
    
    if not db_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials"
        )
    
    # Check if user is OAuth-only (no password)
    if not db_user.password_hash:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="This account uses OAuth authentication. Please sign in with Google."
        )
    
    if not verify_password(user_credentials.password, db_user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials"
        )
    
    if not db_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Account is deactivated"
        )
    
    # Create access token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": db_user.username}, 
        expires_delta=access_token_expires
    )
    
    # Set HTTP-only cookie for better security (optional)
    response.set_cookie(
        key="access_token",
        value=f"Bearer {access_token}",
        httponly=True,
        max_age=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        expires=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )
    
    # Return user data and token
    user_response = UserResponse(
        id=db_user.id,
        username=db_user.username,
        email=db_user.email,
        created_at=db_user.created_at,
        is_active=db_user.is_active,
        oauth_provider=db_user.oauth_provider,
        avatar_url=db_user.avatar_url
    )
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": user_response
    }


# OAuth Routes
@router.get("/oauth/google", response_model=OAuthAuthURL)
async def google_oauth_login():
    """Initiate Google OAuth login"""
    auth_data = oauth_handler.generate_auth_url()
    return OAuthAuthURL(
        auth_url=auth_data['auth_url'],
        state=auth_data['state']
    )


@router.get("/oauth/google/callback")
async def google_oauth_callback(
    code: str = Query(...),
    state: str = Query(...),
    db: Session = Depends(get_db)
):
    """Handle Google OAuth callback"""
    try:
        # Handle OAuth callback
        user_info = await oauth_handler.handle_oauth_callback(code, state)
        
        # Extract user information
        google_id = user_info.get('id')
        email = user_info.get('email')
        name = user_info.get('name', '')
        picture = user_info.get('picture')
        
        if not google_id or not email:
            # Redirect to login with error
            return RedirectResponse(url="/frontend/html/login.html?error=incomplete_user_info")
        
        # Check if user already exists by Google ID
        existing_user = db.query(User).filter(User.google_id == google_id).first()
        
        if existing_user:
            # User exists, update information if needed
            existing_user.avatar_url = picture
            db.commit()
            db_user = existing_user
        else:
            # Check if user exists by email
            existing_email_user = db.query(User).filter(User.email == email).first()
            
            if existing_email_user:
                # Link existing account with Google
                existing_email_user.google_id = google_id
                existing_email_user.oauth_provider = 'google'
                existing_email_user.avatar_url = picture
                db.commit()
                db_user = existing_email_user
            else:
                # Create new user
                # Generate username from name or email
                username = _generate_username_from_name(name, email, db)
                
                db_user = User(
                    username=username,
                    email=email,
                    google_id=google_id,
                    oauth_provider='google',
                    avatar_url=picture,
                    password_hash=None  # OAuth user, no password
                )
                
                db.add(db_user)
                db.commit()
                db.refresh(db_user)
        
        # Create access token
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": db_user.username}, 
            expires_delta=access_token_expires
        )
        
        # Create redirect URL with token as query parameter for frontend to handle
        redirect_url = f"/frontend/html/oauth-success.html?token={access_token}"
        
        return RedirectResponse(url=redirect_url)
        
    except HTTPException as e:
        # Redirect to login with error
        return RedirectResponse(url=f"/frontend/html/login.html?error={e.detail}")
    except Exception as e:
        # Redirect to login with generic error
        return RedirectResponse(url="/frontend/html/login.html?error=oauth_failed")


def _generate_username_from_name(name: str, email: str, db: Session) -> str:
    """Generate a unique username from name or email"""
    # Try to create username from name
    if name:
        # Clean name and make it username-friendly
        base_username = ''.join(c for c in name.lower().replace(' ', '_') if c.isalnum() or c == '_')
    else:
        # Use email prefix
        base_username = email.split('@')[0].lower()
    
    # Ensure it's valid
    if len(base_username) < 3:
        base_username = 'user'
    
    # Make sure it's unique
    username = base_username
    counter = 1
    
    while db.query(User).filter(User.username == username).first():
        username = f"{base_username}_{counter}"
        counter += 1
    
    return username


@router.post("/logout")
async def logout_user(response: Response):
    """Logout user by clearing the cookie"""
    response.delete_cookie(key="access_token")
    return {"message": "Successfully logged out"}


@router.get("/verify")
async def verify_token(current_user: User = Depends(get_current_active_user)):
    """Verify if user is authenticated"""
    return {"message": "Token is valid", "user": current_user.username}


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(current_user: User = Depends(get_current_active_user)):
    """Get current user information"""
    return UserResponse(
        id=current_user.id,
        username=current_user.username,
        email=current_user.email,
        created_at=current_user.created_at,
        is_active=current_user.is_active,
        oauth_provider=current_user.oauth_provider,
        avatar_url=current_user.avatar_url
    ) 