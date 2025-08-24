import hashlib
import json
import logging
from typing import Optional, Dict, List, Any
import redis.asyncio as redis
from src.core.config import settings

logger = logging.getLogger(__name__)


class CacheService:
    """Multi-level caching service for RAG responses"""
    
    def __init__(self):
        self.redis_url = settings.REDIS_URL
        self.ttl_seconds = settings.CACHE_TTL
        self._redis_client = None
    
    async def _get_redis_client(self) -> redis.Redis:
        """Get or create Redis client"""
        if self._redis_client is None:
            try:
                self._redis_client = redis.from_url(self.redis_url)
                # Test connection
                await self._redis_client.ping()
                logger.info("Connected to Redis cache")
            except Exception as e:
                logger.warning(f"Failed to connect to Redis: {e}")
                self._redis_client = None
        return self._redis_client
    
    def generate_cache_key(
        self, 
        query: str, 
        permission_groups: List[str], 
        index_name: str, 
        retriever_config: Dict[str, Any]
    ) -> str:
        """Generate consistent cache key for semantic similarity"""
        # Normalize query for semantic matching
        normalized_query = self._normalize_query(query)
        
        # Create composite key
        key_components = [
            normalized_query,
            sorted(permission_groups),
            index_name,
            sorted(retriever_config.items())
        ]
        
        cache_key = hashlib.md5(
            str(key_components).encode('utf-8')
        ).hexdigest()
        
        return f"rag_cache:{cache_key}"
    
    def _normalize_query(self, query: str) -> str:
        """Normalize query for better cache hits"""
        # Convert to lowercase and remove extra whitespace
        normalized = " ".join(query.lower().strip().split())
        
        # Remove common stop words that don't affect semantics
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        words = [word for word in normalized.split() if word not in stop_words]
        
        return " ".join(words)
    
    async def get_cached_response(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached response with hit rate tracking"""
        try:
            redis_client = await self._get_redis_client()
            if not redis_client:
                return None
                
            cached_data = await redis_client.get(cache_key)
            if cached_data:
                # Track cache hit
                await redis_client.incr("cache_hits")
                logger.info(f"Cache hit for key: {cache_key}")
                return json.loads(cached_data)
        except Exception as e:
            logger.error(f"Cache retrieval error: {e}")
        
        # Track cache miss
        try:
            redis_client = await self._get_redis_client()
            if redis_client:
                await redis_client.incr("cache_misses")
        except:
            pass
            
        return None
    
    async def cache_response(self, cache_key: str, response: Dict[str, Any]) -> None:
        """Cache response with TTL"""
        try:
            redis_client = await self._get_redis_client()
            if not redis_client:
                return
                
            # Remove trace_id from cached response to avoid conflicts
            cached_response = response.copy()
            cached_response.pop("trace_id", None)
            
            await redis_client.setex(
                cache_key,
                self.ttl_seconds,
                json.dumps(cached_response, ensure_ascii=False, default=str)
            )
            logger.info(f"Cached response for key: {cache_key}")
        except Exception as e:
            logger.error(f"Cache storage error: {e}")

    async def get_cached_result(self, cache_key: str) -> Optional[Any]:
        """Generic getter for cached values (dict, list, str, etc.)."""
        try:
            redis_client = await self._get_redis_client()
            if not redis_client:
                return None
            cached_data = await redis_client.get(cache_key)
            if cached_data:
                try:
                    return json.loads(cached_data)
                except Exception:
                    return None
        except Exception as e:
            logger.error(f"Cache retrieval error (generic): {e}")
        return None

    async def cache_result(self, cache_key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Generic setter for cached values with optional TTL override."""
        try:
            redis_client = await self._get_redis_client()
            if not redis_client:
                return
            await redis_client.setex(
                cache_key,
                ttl if ttl is not None else self.ttl_seconds,
                json.dumps(value, ensure_ascii=False, default=str)
            )
            logger.info(f"Cached value for key: {cache_key}")
        except Exception as e:
            logger.error(f"Cache storage error (generic): {e}")
    
    async def invalidate_cache_pattern(self, pattern: str) -> int:
        """Invalidate cache entries matching pattern"""
        try:
            redis_client = await self._get_redis_client()
            if not redis_client:
                return 0
                
            keys = await redis_client.keys(pattern)
            if keys:
                deleted = await redis_client.delete(*keys)
                logger.info(f"Invalidated {deleted} cache entries matching pattern: {pattern}")
                return deleted
        except Exception as e:
            logger.error(f"Cache invalidation error: {e}")
        
        return 0
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        try:
            redis_client = await self._get_redis_client()
            if not redis_client:
                return {"status": "unavailable"}
            
            hits = await redis_client.get("cache_hits") or 0
            misses = await redis_client.get("cache_misses") or 0
            
            hits = int(hits)
            misses = int(misses)
            total = hits + misses
            
            hit_rate = (hits / total * 100) if total > 0 else 0
            
            # Get Redis info
            info = await redis_client.info()
            
            return {
                "status": "available",
                "hits": hits,
                "misses": misses,
                "hit_rate": round(hit_rate, 2),
                "total_requests": total,
                "memory_usage": info.get("used_memory_human", "unknown"),
                "connected_clients": info.get("connected_clients", 0)
            }
        except Exception as e:
            logger.error(f"Cache stats error: {e}")
            return {"status": "error", "error": str(e)}
    
    async def clear_all_cache(self) -> bool:
        """Clear all RAG cache entries"""
        try:
            redis_client = await self._get_redis_client()
            if not redis_client:
                return False
                
            # Clear only RAG-related cache
            deleted = await self.invalidate_cache_pattern("rag_cache:*")
            
            # Reset counters
            await redis_client.delete("cache_hits", "cache_misses")
            
            logger.info(f"Cleared all RAG cache ({deleted} entries)")
            return True
        except Exception as e:
            logger.error(f"Cache clear error: {e}")
            return False
    
    async def close(self):
        """Close Redis connection"""
        if self._redis_client:
            await self._redis_client.close()
            logger.info("Redis connection closed")
