"""A small, local enterprise-agent test bench."""

from .config import LabConfig, load_config
from .models import AgentDecision, RunTrace

__all__ = ["AgentDecision", "LabConfig", "RunTrace", "load_config"]

__version__ = "0.1.0"
