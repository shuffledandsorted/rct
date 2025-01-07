from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class AgentConfig:
    """Configuration for quantum agents."""
    critical_depth: int = 5
    memory_size: int = 10
    coherence_threshold: float = 0.6
    detection_threshold: float = 0.5
    max_children: int = 4
    decay_rate: float = 0.1

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "AgentConfig":
        """Create config from dictionary, using defaults for missing values"""
        return cls(**{
            k: v for k, v in config_dict.items() 
            if k in cls.__dataclass_fields__
        }) 
