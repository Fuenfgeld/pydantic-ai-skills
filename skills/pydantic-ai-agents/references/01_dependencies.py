"""
Example: Dependencies (Dependency Injection Pattern)

This file demonstrates the Golden Standard for managing state and configuration
in Pydantic AI agents using the Dependency Injection pattern.
"""

from dataclasses import dataclass
from typing import Optional
import httpx


@dataclass
class AgentDeps:
    """
    The Dependency Container.

    This class holds all runtime state, configuration, and connections
    required by the agent. It is injected into every tool and prompt.

    Key Principles:
    - Use dataclasses for clean, type-safe dependency containers
    - Never use global variables for state
    - Include all external connections (HTTP clients, DB connections, etc.)
    - Add helper methods for common checks or transformations
    """
    user_id: str
    api_key: str
    http_client: httpx.AsyncClient
    environment: str = "production"

    # Helper methods can be added to encapsulate logic
    def is_dev(self) -> bool:
        """Check if running in development environment."""
        return self.environment == "development"

    def get_auth_header(self) -> dict[str, str]:
        """Generate authentication header for API requests."""
        return {"Authorization": f"Bearer {self.api_key}"}


# Usage Pattern:
# 1. Define your dependency container with all required state
# 2. Pass it to the Agent using deps_type parameter
# 3. Access it in tools and prompts via ctx.deps
