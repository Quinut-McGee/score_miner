"""
Persistent HTTP connection pool to eliminate connection setup latency.

Instead of creating a new connection for each video (2.4s overhead),
keep connections warm and reuse them (near-zero overhead).
"""

import httpx
import asyncio
from typing import Optional
from loguru import logger


class ConnectionPool:
    """Singleton HTTP client with persistent connections."""
    
    _instance: Optional['ConnectionPool'] = None
    _client: Optional[httpx.AsyncClient] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._client is None:
            self._initialize_client()
    
    def _initialize_client(self):
        """Initialize persistent HTTP client with optimal settings."""
        limits = httpx.Limits(
            max_connections=20,
            max_keepalive_connections=10,
            keepalive_expiry=30.0  # Keep connections alive for 30s
        )
        
        timeout = httpx.Timeout(
            connect=2.0,    # 2s to establish connection
            read=30.0,      # 30s to read response
            write=10.0,     # 10s to send request
            pool=1.0        # 1s to get connection from pool
        )
        
        # Try HTTP/2 if available
        try:
            import h2  # noqa: F401
            http2_enabled = True
            logger.info("HTTP/2 support enabled for connection pool")
        except ImportError:
            http2_enabled = False
            logger.warning("HTTP/2 not available (install: pip install httpx[http2])")
        
        self._client = httpx.AsyncClient(
            timeout=timeout,
            limits=limits,
            http2=http2_enabled,
            follow_redirects=True,
        )
        
        logger.info(f"Connection pool initialized (HTTP/2: {http2_enabled})")
    
    @property
    def client(self) -> httpx.AsyncClient:
        """Get the persistent HTTP client."""
        if self._client is None:
            self._initialize_client()
        return self._client
    
    async def close(self):
        """Close the connection pool."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
            logger.info("Connection pool closed")
    
    async def warmup(self, url: str):
        """Pre-warm connection to a specific host."""
        try:
            # Make a quick HEAD request to establish connection
            await self.client.head(url, timeout=5.0)
            logger.info(f"Connection warmed up to {url}")
        except Exception as e:
            logger.warning(f"Failed to warm up connection: {e}")


# Global singleton instance
_pool = ConnectionPool()


def get_pool() -> ConnectionPool:
    """Get the global connection pool instance."""
    return _pool


async def close_pool():
    """Close the global connection pool."""
    await _pool.close()

