from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class AgentConfig:
    """Configuration parameters for quantum agents"""
    # Detection parameters
    detection_threshold: float = 0.5
    spawn_threshold: float = 0.8
    coherence_threshold: float = 0.7
    
    # Memory parameters
    memory_size: int = 5
    decay_rate: float = 0.1
    
    # Feature detection parameters
    min_feature_strength: float = 0.3
    max_children: int = 5
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "AgentConfig":
        """Create config from dictionary, using defaults for missing values"""
        return cls(**{
            k: v for k, v in config_dict.items() 
            if k in cls.__dataclass_fields__
        }) 