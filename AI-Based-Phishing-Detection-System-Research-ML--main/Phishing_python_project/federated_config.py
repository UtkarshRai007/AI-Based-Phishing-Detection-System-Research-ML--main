"""
Configuration file for Federated Learning and Differential Privacy settings
"""

import os
from dataclasses import dataclass
from typing import Dict, Any, Optional

@dataclass
class FederatedConfig:
    """Configuration for federated learning system"""
    
    # Federated Learning Settings
    num_clients: int = 5
    num_rounds: int = 10
    min_clients_per_round: int = 2
    aggregation_strategy: str = "fedavg"  # fedavg, fedprox, fednova
    
    # Training Settings
    local_epochs: int = 5
    batch_size: int = 32
    learning_rate: float = 0.001
    momentum: float = 0.9
    
    # Model Settings
    model_type: str = "neural_network"  # neural_network, xgboost, random_forest
    hidden_size: int = 128
    dropout_rate: float = 0.3
    
    # Differential Privacy Settings
    use_differential_privacy: bool = True
    epsilon: float = 1.0  # Privacy budget
    delta: float = 1e-5   # Privacy failure probability
    noise_multiplier: float = 1.1
    max_grad_norm: float = 1.0
    
    # Security Settings
    use_secure_aggregation: bool = True
    encryption_key_size: int = 256
    use_hmac_verification: bool = True
    
    # Communication Settings
    timeout: int = 300  # seconds
    max_retries: int = 3
    heartbeat_interval: int = 30  # seconds
    
    # Logging and Monitoring
    log_level: str = "INFO"
    save_model_checkpoints: bool = True
    checkpoint_interval: int = 5  # rounds
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            'num_clients': self.num_clients,
            'num_rounds': self.num_rounds,
            'min_clients_per_round': self.min_clients_per_round,
            'aggregation_strategy': self.aggregation_strategy,
            'local_epochs': self.local_epochs,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'momentum': self.momentum,
            'model_type': self.model_type,
            'hidden_size': self.hidden_size,
            'dropout_rate': self.dropout_rate,
            'use_differential_privacy': self.use_differential_privacy,
            'epsilon': self.epsilon,
            'delta': self.delta,
            'noise_multiplier': self.noise_multiplier,
            'max_grad_norm': self.max_grad_norm,
            'use_secure_aggregation': self.use_secure_aggregation,
            'encryption_key_size': self.encryption_key_size,
            'use_hmac_verification': self.use_hmac_verification,
            'timeout': self.timeout,
            'max_retries': self.max_retries,
            'heartbeat_interval': self.heartbeat_interval,
            'log_level': self.log_level,
            'save_model_checkpoints': self.save_model_checkpoints,
            'checkpoint_interval': self.checkpoint_interval
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'FederatedConfig':
        """Create configuration from dictionary"""
        return cls(**config_dict)
    
    def validate(self) -> bool:
        """Validate configuration parameters"""
        errors = []
        
        if self.num_clients < 2:
            errors.append("Number of clients must be at least 2")
        
        if self.num_rounds < 1:
            errors.append("Number of rounds must be at least 1")
        
        if self.min_clients_per_round > self.num_clients:
            errors.append("Minimum clients per round cannot exceed total clients")
        
        if self.epsilon <= 0:
            errors.append("Epsilon must be positive")
        
        if self.delta <= 0 or self.delta >= 1:
            errors.append("Delta must be between 0 and 1")
        
        if self.noise_multiplier <= 0:
            errors.append("Noise multiplier must be positive")
        
        if self.learning_rate <= 0:
            errors.append("Learning rate must be positive")
        
        if self.batch_size < 1:
            errors.append("Batch size must be positive")
        
        if errors:
            raise ValueError(f"Configuration validation failed: {'; '.join(errors)}")
        
        return True

@dataclass
class PrivacyConfig:
    """Configuration for differential privacy"""
    
    epsilon: float = 1.0
    delta: float = 1e-5
    noise_multiplier: float = 1.1
    max_grad_norm: float = 1.0
    sample_rate: Optional[float] = None
    target_epsilon: Optional[float] = None
    target_delta: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'epsilon': self.epsilon,
            'delta': self.delta,
            'noise_multiplier': self.noise_multiplier,
            'max_grad_norm': self.max_grad_norm,
            'sample_rate': self.sample_rate,
            'target_epsilon': self.target_epsilon,
            'target_delta': self.target_delta
        }

@dataclass
class SecurityConfig:
    """Configuration for security features"""
    
    use_encryption: bool = True
    encryption_algorithm: str = "AES-256"
    key_size: int = 256
    use_hmac: bool = True
    hmac_algorithm: str = "SHA-256"
    use_ssl: bool = True
    certificate_path: Optional[str] = None
    private_key_path: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'use_encryption': self.use_encryption,
            'encryption_algorithm': self.encryption_algorithm,
            'key_size': self.key_size,
            'use_hmac': self.use_hmac,
            'hmac_algorithm': self.hmac_algorithm,
            'use_ssl': self.use_ssl,
            'certificate_path': self.certificate_path,
            'private_key_path': self.private_key_path
        }

# Default configurations
DEFAULT_FEDERATED_CONFIG = FederatedConfig()
DEFAULT_PRIVACY_CONFIG = PrivacyConfig()
DEFAULT_SECURITY_CONFIG = SecurityConfig()

# Environment-based configuration
def get_config_from_env() -> FederatedConfig:
    """Get configuration from environment variables"""
    config = FederatedConfig()
    
    # Override with environment variables if present
    if os.getenv('FED_NUM_CLIENTS'):
        config.num_clients = int(os.getenv('FED_NUM_CLIENTS'))
    
    if os.getenv('FED_NUM_ROUNDS'):
        config.num_rounds = int(os.getenv('FED_NUM_ROUNDS'))
    
    if os.getenv('FED_EPSILON'):
        config.epsilon = float(os.getenv('FED_EPSILON'))
    
    if os.getenv('FED_DELTA'):
        config.delta = float(os.getenv('FED_DELTA'))
    
    if os.getenv('FED_NOISE_MULTIPLIER'):
        config.noise_multiplier = float(os.getenv('FED_NOISE_MULTIPLIER'))
    
    if os.getenv('FED_LEARNING_RATE'):
        config.learning_rate = float(os.getenv('FED_LEARNING_RATE'))
    
    if os.getenv('FED_BATCH_SIZE'):
        config.batch_size = int(os.getenv('FED_BATCH_SIZE'))
    
    return config

# Configuration presets
PRESET_CONFIGS = {
    'high_privacy': FederatedConfig(
        epsilon=0.5,
        delta=1e-6,
        noise_multiplier=2.0,
        num_rounds=15
    ),
    'balanced': FederatedConfig(
        epsilon=1.0,
        delta=1e-5,
        noise_multiplier=1.1,
        num_rounds=10
    ),
    'high_accuracy': FederatedConfig(
        epsilon=2.0,
        delta=1e-4,
        noise_multiplier=0.8,
        num_rounds=8,
        local_epochs=8
    ),
    'fast_training': FederatedConfig(
        epsilon=1.5,
        delta=1e-4,
        noise_multiplier=1.0,
        num_rounds=5,
        local_epochs=3
    )
}

def get_preset_config(preset_name: str) -> FederatedConfig:
    """Get a preset configuration"""
    if preset_name not in PRESET_CONFIGS:
        raise ValueError(f"Unknown preset: {preset_name}. Available: {list(PRESET_CONFIGS.keys())}")
    return PRESET_CONFIGS[preset_name]

# Utility functions
def calculate_privacy_budget(epsilon: float, delta: float, rounds: int) -> Dict[str, float]:
    """Calculate privacy budget consumption"""
    # Simple privacy composition (can be improved with advanced composition theorems)
    total_epsilon = epsilon * rounds
    total_delta = delta * rounds
    
    return {
        'total_epsilon': total_epsilon,
        'total_delta': total_delta,
        'epsilon_per_round': epsilon,
        'delta_per_round': delta,
        'rounds': rounds
    }

def estimate_noise_multiplier(epsilon: float, delta: float, sample_rate: float) -> float:
    """Estimate noise multiplier for given privacy parameters"""
    # Simplified estimation based on Gaussian mechanism
    import math
    c = math.sqrt(2 * math.log(1.25 / delta))
    noise_multiplier = c / (epsilon * sample_rate)
    return max(noise_multiplier, 0.1)  # Ensure minimum noise

def validate_privacy_parameters(epsilon: float, delta: float) -> bool:
    """Validate privacy parameters"""
    if epsilon <= 0:
        return False, "Epsilon must be positive"
    
    if delta <= 0 or delta >= 1:
        return False, "Delta must be between 0 and 1"
    
    if epsilon > 10:
        return False, "Epsilon should typically be less than 10 for meaningful privacy"
    
    if delta > 1e-3:
        return False, "Delta should typically be less than 0.001 for meaningful privacy"
    
    return True, "Parameters are valid"
