"""
Pydantic schemas for authentication request/response validation.
"""

from pydantic import BaseModel, EmailStr, field_validator
from typing import Optional
from datetime import datetime


class UserBase(BaseModel):
    """Base user schema"""
    username: str
    email: EmailStr


class UserCreate(UserBase):
    """User creation schema"""
    password: str
    confirm_password: str

    @field_validator('username')
    @classmethod
    def validate_username(cls, v):
        if len(v) < 3:
            raise ValueError('Username must be at least 3 characters long')
        if len(v) > 50:
            raise ValueError('Username must be less than 50 characters')
        if not v.replace('_', '').isalnum():
            raise ValueError('Username can only contain letters, numbers, and underscores')
        return v

    @field_validator('password')
    @classmethod
    def validate_password(cls, v):
        if len(v) < 6:
            raise ValueError('Password must be at least 6 characters long')
        return v

    @field_validator('confirm_password')
    @classmethod
    def passwords_match(cls, v, info):
        if 'password' in info.data and v != info.data['password']:
            raise ValueError('Passwords do not match')
        return v


class UserLogin(BaseModel):
    """User login schema"""
    username: str  # Can be username or email
    password: str


class UserResponse(UserBase):
    """User response schema"""
    id: int
    created_at: datetime
    is_active: bool
    oauth_provider: Optional[str] = None
    avatar_url: Optional[str] = None

    class Config:
        from_attributes = True


class Token(BaseModel):
    """Token response schema"""
    access_token: str
    token_type: str
    user: UserResponse


class TokenData(BaseModel):
    """Token data schema"""
    username: Optional[str] = None


class OAuthAuthURL(BaseModel):
    """OAuth authorization URL response"""
    auth_url: str
    state: str


class OAuthCallback(BaseModel):
    """OAuth callback data"""
    code: str
    state: str 