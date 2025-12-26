# agents/application/services/agent_cache.py

import hashlib
import hmac
import logging
import os
from typing import Optional

import dill
import redis
from cachetools import LRUCache

from domain.template import Template as DomainTemplate

logger = logging.getLogger(__name__)

class PipelineCache:
    def __init__(
        self,
        lru_maxsize: int = 128,
        ttl: int = 3600,
        secret_key: str | None = None,
        enable_hmac: bool = True,
    ):
        self._lru_cache = LRUCache(maxsize=lru_maxsize)
        self._ttl = ttl

        self._enable_hmac = bool(enable_hmac and secret_key)
        if enable_hmac and not secret_key:
            logger.warning(
                "PipelineCache: secret_key not provided; HMAC signing disabled (dev only)."
            )

        self._secret_key = secret_key.encode("utf-8") if self._enable_hmac else b""
        self._signature_len = hashlib.sha256().digest_size if self._enable_hmac else 0
        self._key_prefix = "pipeline_template:hmac" if self._enable_hmac else "pipeline_template:plain"

        self._redis = None
        self._connect_to_redis()

        logger.info("PipelineCache (dill%s) initialized.", "+HMAC" if self._enable_hmac else "")

    def _get_redis_key(self, pipeline_id: str) -> str:
        return f"{self._key_prefix}:{pipeline_id}"

    def get(self, pipeline_id: str) -> Optional[DomainTemplate]:
        if pipeline_id in self._lru_cache:
            return self._lru_cache[pipeline_id]

        if not self._redis:
            return None

        try:
            payload = self._redis.get(self._get_redis_key(pipeline_id))
            if not payload:
                return None

            if self._enable_hmac:
                serialized_dto = payload[: -self._signature_len]
                stored_signature = payload[-self._signature_len :]
                expected_signature = hmac.new(
                    self._secret_key, serialized_dto, hashlib.sha256
                ).digest()

                if not hmac.compare_digest(expected_signature, stored_signature):
                    logger.warning(f"HMAC verification failed for pipeline_id: {pipeline_id}.")
                    return None
            else:
                serialized_dto = payload

            hydrated_template = dill.loads(serialized_dto)
            self._lru_cache[pipeline_id] = hydrated_template
            return hydrated_template

        except (redis.RedisError, dill.UnpicklingError) as e:
            logger.error(
                f"Failed to get template for pipeline {pipeline_id} from Redis: {e}",
                exc_info=True,
            )
            return None

    def add(self, template: DomainTemplate):
        pipeline_id_str = str(template.id)
        self._lru_cache[pipeline_id_str] = template

        if not self._redis:
            return

        try:
            serialized_dto = dill.dumps(template)

            if self._enable_hmac:
                signature = hmac.new(
                    self._secret_key, serialized_dto, hashlib.sha256
                ).digest()
                payload = serialized_dto + signature
            else:
                payload = serialized_dto

            self._redis.set(self._get_redis_key(pipeline_id_str), payload, ex=self._ttl)

        except (redis.RedisError, dill.PicklingError) as e:
            logger.error(
                f"Failed to add template for pipeline {pipeline_id_str} to Redis: {e}",
                exc_info=True,
            )
