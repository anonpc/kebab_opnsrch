"""
Authentication and authorization endpoints.
Simplified stub implementation - security is disabled at current stage.
"""

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel
import logging

router = APIRouter()
logger = logging.getLogger("auth")


class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int


@router.post("/token", response_model=TokenResponse)
async def login(request: LoginRequest):
    """
    Authentication endpoint - currently disabled/stubbed.
    Returns a dummy token for development purposes.
    """
    logger.info(f"Authentication attempt for user: {request.username}")
    
    # For development, return a dummy token
    return TokenResponse(
        access_token="dummy-token-for-development",
        expires_in=3600
    )


@router.post("/verify")
async def verify_token():
    """
    Token verification endpoint - currently disabled/stubbed.
    """
    return {
        "valid": True,
        "username": "development-user",
        "message": "Authentication is disabled in development mode"
    }


@router.get("/status")
async def auth_status():
    """
    Authentication status endpoint.
    """
    return {
        "authentication_enabled": False,
        "mode": "development",
        "message": "Authentication is disabled at current stage"
    }
