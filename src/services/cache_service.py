import hashlib
import json
import logging
from typing import Optional, Dict, List, Any
import redis.asyncio as redis
from src.core.config import settings

logger = logging.getLogger(__name__)


class CacheService:
    """Multi-level caching service for RAG responses

    초보자용 설명:
    - Redis에 질의/중간 결과/최종 응답을 저장해 다음 요청에서 재사용합니다.
    - TTL(만료 시간)이 지나면 자동으로 삭제됩니다.
    """
    
    def __init__(self):
        self.redis_url = settings.REDIS_URL
        self.ttl_seconds = settings.CACHE_TTL
        self._redis_client = None
    
    async def _get_redis_client(self) -> redis.Redis:
        """Get or create Redis client

        초보자용 설명:
        - 첫 호출 때만 Redis 클라이언트를 만들고, 이후에는 재사용합니다.
        """
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
        """Generate consistent cache key for semantic similarity

        초보자용 설명:
        - 질의/권한/인덱스/설정의 조합을 바탕으로 고유한 캐시 키를 만듭니다.
        """
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
        """Normalize query for better cache hits

        초보자용 설명:
        - 불필요한 공백/대소문자/자주 등장하는 관사 등을 제거해 캐시 적중률을 높입니다.
        """
        # Convert to lowercase and remove extra whitespace
        normalized = " ".join(query.lower().strip().split())
        
        # Remove common stop words that don't affect semantics
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        words = [word for word in normalized.split() if word not in stop_words]
        
        return " ".join(words)
    
    async def get_cached_response(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached response with hit rate tracking

        초보자용 설명:
        - 캐시에서 응답을 가져오고, 히트/미스 카운트를 기록합니다.
        """
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
        """Cache response with TTL

        초보자용 설명:
        - trace_id는 매 요청마다 달라지므로 캐시에 저장하기 전에 제거합니다.
        """
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
        """Generic getter for cached values (dict, list, str, etc.).

        초보자용 설명:
        - 중간 결과(배열/문자열 등)도 쉽게 가져올 수 있게 만든 범용 메서드입니다.
        """
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
        """Generic setter for cached values with optional TTL override.

        초보자용 설명:
        - 중간 결과를 저장하는 범용 메서드입니다. TTL을 개별 지정할 수 있습니다.
        """
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
        """Invalidate cache entries matching pattern

        초보자용 설명:
        - 주어진 패턴과 일치하는 키를 찾아 삭제합니다.
        """
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
        """Get cache performance statistics

        초보자용 설명:
        - 캐시 히트율/메모리 사용량/클라이언트 수 등의 정보를 확인합니다.
        """
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
        """Clear all RAG cache entries

        초보자용 설명:
        - RAG 관련 키만 제거하고, 카운터도 초기화합니다.
        """
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
        """Close Redis connection

        초보자용 설명:
        - 애플리케이션 종료 시 Redis 연결을 정리합니다.
        """
        if self._redis_client:
            await self._redis_client.close()
            logger.info("Redis connection closed")
