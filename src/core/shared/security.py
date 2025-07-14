"""
Security management and authentication system.
Simplified stub version - security is disabled at current stage.
"""

import logging
from typing import Optional, Dict, Any, List
from datetime import datetime
from pydantic import BaseModel
from fastapi import Request

logger = logging.getLogger(__name__)


class User(BaseModel):
    """User model for authentication."""
    username: str
    email: Optional[str] = None
    is_active: bool = True
    is_admin: bool = False
    roles: List[str] = ["user"]  # Default role, can include "admin"


class SecurityConfig(BaseModel):
    """Security configuration - disabled for development."""
    enable_auth: bool = False
    jwt_secret: str = "development-secret"
    jwt_expiration_hours: int = 24


class AuthenticationError(Exception):
    """Authentication error."""
    pass


class AuthManager:
    """Authentication manager - stub implementation."""
    
    def authenticate_user(self, username: str, password: str) -> Optional[User]:
        """Authenticate user - stub implementation."""
        logger.info(f"Development mode authentication for user: {username}")
        return User(
            username=username, 
            email=f"{username}@example.com",
            is_admin=True,
            roles=["user", "admin"]  # Give admin rights in dev mode
        )
    
    def create_access_token(self, username: str) -> str:
        """Create access token - stub implementation."""
        return f"dummy-token-{username}-{datetime.now().timestamp()}"
    
    def verify_token(self, token: str) -> Dict[str, Any]:
        """Verify token - stub implementation."""
        return {
            "sub": "development-user",
            "exp": datetime.now().timestamp() + 3600
        }
    
    def get_user_by_username(self, username: str) -> Optional[User]:
        """Get user by username - stub implementation."""
        return User(
            username=username,
            is_admin=True,
            roles=["user", "admin"]  # Give admin rights in dev mode
        )


class RateLimiter:
    """Rate limiter - stub implementation."""
    
    def is_allowed(self, client_ip: str) -> bool:
        """Check if request is allowed - always allow in dev mode."""
        return True


class InputValidator:
    """Input validator - stub implementation."""
    
    def validate_search_query(self, query: str) -> str:
        """Validate search query - basic cleanup in dev mode."""
        if not query or not query.strip():
            raise ValueError("Search query cannot be empty")
        
        # Basic sanitization
        cleaned = query.strip()
        if len(cleaned) > 1000:
            cleaned = cleaned[:1000]
        
        return cleaned


class SecurityManager:
    """Security manager - stub implementation."""
    
    def __init__(self):
        self.security_config = SecurityConfig()
        self.auth_manager = AuthManager()
        self.rate_limiter = RateLimiter()
        self.input_validator = InputValidator()
    
    def get_client_ip(self, request: Request) -> str:
        """Get client IP address from request."""
        # Check for forwarded headers first
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            # Take the first IP in the chain
            return forwarded_for.split(',')[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip.strip()
        
        # Fallback to client IP
        if hasattr(request, 'client') and request.client:
            return request.client.host
        
        return "127.0.0.1"  # Default fallback


def get_current_user() -> User:
    """Get current user - stub implementation with admin rights in dev mode."""
    return User(
        username="development-user",
        email="dev@example.com",
        is_admin=True,
        roles=["user", "admin"]  # Give admin rights in dev mode
    )


# Global security manager instance (stub)
_security_manager: Optional[SecurityManager] = None


def get_security_manager() -> SecurityManager:
    """Get global security manager instance."""
    global _security_manager
    
    if _security_manager is None:
        _security_manager = SecurityManager()
    
    return _security_manager
