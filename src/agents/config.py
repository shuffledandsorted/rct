from dataclasses import dataclass
from typing import Dict, Any


class AgentConfig:
    def __init__(
        self,
        critical_depth: int = 7,
        memory_size: int = 10,
        coherence_threshold: float = 0.8,
        personality: str = "balanced"
    ):
        self.critical_depth = critical_depth
        self.memory_size = memory_size
        self.coherence_threshold = coherence_threshold
        self.personality = personality
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "AgentConfig":
        """Create config from dictionary, using defaults for missing values"""
        return cls(**{
            k: v for k, v in config_dict.items() 
            if k in cls.__dataclass_fields__
        }) 