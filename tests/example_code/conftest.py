"""
Example Code test fixtures.

Provides fixtures specific to testing the exampleCode examples.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock


@pytest.fixture
def mock_openrouter_response():
    """Mock response from OpenRouter API."""
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = "Test response"
    return response
