"""
Simple safety module for sheily_core
"""
from typing import Tuple

class SecurityMonitor:
    def check_request(self, query: str, client_id: str) -> Tuple[bool, str]:
        """Check if request is safe"""
        # Basic safety check
        unsafe_keywords = ["rm -rf", "drop table", "shutdown"]
        for kw in unsafe_keywords:
            if kw in query.lower():
                return False, f"Unsafe keyword detected: {kw}"
        return True, "safe"

def get_security_monitor() -> SecurityMonitor:
    return SecurityMonitor()
