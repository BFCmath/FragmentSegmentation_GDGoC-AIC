"""
OAuth handler for Google authentication.
"""

import json
import secrets
from typing import Optional, Dict, Any
from urllib.parse import urlencode

import httpx
from authlib.integrations.starlette_client import OAuth
from fastapi import HTTPException, status

import config as cfg


class OAuthHandler:
    """Handles OAuth authentication flows"""
    
    def __init__(self):
        self.oauth = OAuth()
        self.oauth.register(
            name='google',
            client_id=cfg.GOOGLE_CLIENT_ID,
            client_secret=cfg.GOOGLE_CLIENT_SECRET,
            server_metadata_url=cfg.GOOGLE_DISCOVERY_URL,
            client_kwargs={
                'scope': ' '.join(cfg.GOOGLE_SCOPES)
            }
        )
        
        # Store state for CSRF protection
        self._states = {}
    
    def generate_auth_url(self) -> Dict[str, str]:
        """Generate Google OAuth authorization URL"""
        if not cfg.GOOGLE_CLIENT_ID or not cfg.GOOGLE_CLIENT_SECRET:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="OAuth not configured. Please contact administrator."
            )
        
        # Generate CSRF state
        state = secrets.token_urlsafe(32)
        
        # Store state (in production, use Redis or database)
        self._states[state] = True
        
        # Build authorization URL
        params = {
            'client_id': cfg.GOOGLE_CLIENT_ID,
            'redirect_uri': cfg.GOOGLE_REDIRECT_URI,
            'scope': ' '.join(cfg.GOOGLE_SCOPES),
            'response_type': 'code',
            'state': state,
            'access_type': 'offline',
            'prompt': 'consent'
        }
        
        auth_url = f"https://accounts.google.com/o/oauth2/auth?{urlencode(params)}"
        
        return {
            'auth_url': auth_url,
            'state': state
        }
    
    async def handle_oauth_callback(self, code: str, state: str) -> Dict[str, Any]:
        """Handle OAuth callback and extract user information"""
        # Verify state for CSRF protection
        if state not in self._states:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid state parameter"
            )
        
        # Remove used state
        del self._states[state]
        
        try:
            # Exchange code for access token
            token_data = await self._exchange_code_for_token(code)
            
            # Get user information
            user_info = await self._get_user_info(token_data['access_token'])
            
            return user_info
            
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"OAuth authentication failed: {str(e)}"
            )
    
    async def _exchange_code_for_token(self, code: str) -> Dict[str, Any]:
        """Exchange authorization code for access token"""
        token_url = "https://oauth2.googleapis.com/token"
        
        data = {
            'client_id': cfg.GOOGLE_CLIENT_ID,
            'client_secret': cfg.GOOGLE_CLIENT_SECRET,
            'code': code,
            'grant_type': 'authorization_code',
            'redirect_uri': cfg.GOOGLE_REDIRECT_URI
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(token_url, data=data)
            response.raise_for_status()
            return response.json()
    
    async def _get_user_info(self, access_token: str) -> Dict[str, Any]:
        """Get user information from Google"""
        user_info_url = "https://www.googleapis.com/oauth2/v2/userinfo"
        
        headers = {
            'Authorization': f'Bearer {access_token}'
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.get(user_info_url, headers=headers)
            response.raise_for_status()
            return response.json()
    
    def validate_state(self, state: str) -> bool:
        """Validate OAuth state parameter"""
        return state in self._states


# Global OAuth handler instance
oauth_handler = OAuthHandler() 