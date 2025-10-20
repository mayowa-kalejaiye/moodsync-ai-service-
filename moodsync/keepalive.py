import threading
import time
import logging
from typing import Dict
import requests

from .resilience import intelligent_retry, RetryConfig, AIServiceError

logger = logging.getLogger(__name__)


class KeepAliveService:
    def __init__(self, base_url: str = None, interval: int = 600):
        self.base_url = base_url
        self.interval = interval
        self.running = False
        self.thread = None
        self.logger = logging.getLogger(__name__ + '.KeepAlive')

    def start(self):
        if not self.base_url:
            self.logger.warning("Keep-alive service: No base URL provided, cannot start")
            return
        if self.running:
            self.logger.warning("Keep-alive service already running")
            return
        self.running = True
        self.thread = threading.Thread(target=self._keep_alive_loop, daemon=True)
        self.thread.start()
        self.logger.info(f"Keep-alive service started - pinging {self.base_url}/api/health/ every {self.interval//60} minutes")

    def stop(self):
        self.running = False
        if self.thread:
            self.logger.info("Keep-alive service stopped")

    @intelligent_retry(config=RetryConfig())
    def _ping_health_endpoint_core(self, health_url: str) -> bool:
        try:
            response = requests.get(health_url, timeout=10)
            if response.status_code == 200:
                self.logger.debug(f"Keep-alive ping successful: {health_url}")
                return True
            else:
                raise AIServiceError(f"Health check failed with status {response.status_code}", error_type="http_error", is_retryable=True)
        except requests.RequestException as e:
            raise AIServiceError(f"Network error during health check: {str(e)}", error_type="network", is_retryable=True)

    def _ping_health_endpoint(self):
        health_url = f"{self.base_url}/api/health/"
        try:
            self._ping_health_endpoint_core(health_url)
        except AIServiceError as e:
            self.logger.error(f"Keep-alive ping failed after all retries: {e.error_type} - {str(e)}")
        except Exception as e:
            self.logger.error(f"Unexpected error in keep-alive ping: {str(e)}")

    def _keep_alive_loop(self):
        while self.running:
            try:
                time.sleep(self.interval)
                if not self.running:
                    break
                self._ping_health_endpoint()
            except Exception as e:
                self.logger.error(f"Keep-alive service error: {str(e)}")
