import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager
import flwr as fl
from typing import Dict, List, Tuple, Optional
import json
import os
from cryptography.fernet import Fernet
import hashlib
import hmac
import secrets
from collections import OrderedDict

class PhishingNeuralNetwork(nn.Module):
    """Neural network for phishing detection with privacy guarantees"""
    
    def __init__(self, input_size: int, hidden_size: int = 128, num_classes: int = 2):
        super(PhishingNeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, num_classes)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

class DifferentialPrivacyConfig:
    """Configuration for differential privacy parameters"""
    
    def __init__(self, epsilon: float = 1.0, delta: float = 1e-5, noise_multiplier: float = 1.1):
        self.epsilon = epsilon  # Privacy budget
        self.delta = delta      # Privacy failure probability
        self.noise_multiplier = noise_multiplier  # Noise scale
        
    def to_dict(self):
        return {
            'epsilon': self.epsilon,
            'delta': self.delta,
            'noise_multiplier': self.noise_multiplier
        }

class FederatedPhishingClient(fl.client.NumPyClient):
    """Federated Learning client for phishing detection with differential privacy"""
    
    def __init__(self, 
                 client_id: str,
                 data: pd.DataFrame,
                 privacy_config: DifferentialPrivacyConfig,
                 model_config: Dict = None):
        self.client_id = client_id
        self.data = data
        self.privacy_config = privacy_config
        self.model_config = model_config or {}
        
        # Initialize model and data
        self._prepare_data()
        self._initialize_model()
        self._setup_privacy_engine()
        
        # Security and privacy tracking
        self.privacy_consumed = 0.0
        self.epochs_trained = 0
        
    def _prepare_data(self):
        """Prepare data for training"""
        # Extract features and labels
        features = self._extract_features(self.data['url'])
        labels = self.data['label'].values
        
        # Convert to tensors
        self.X = torch.FloatTensor(features)
        self.y = torch.LongTensor(labels)
        
        # Create data loader
        dataset = TensorDataset(self.X, self.y)
        self.train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
        
    def _extract_features(self, urls):
        """Extract features from URLs for neural network"""
        features = []
        for url in urls:
            feature_vector = self._extract_single_url_features(url)
            features.append(feature_vector)
        return np.array(features)
    
    def _extract_single_url_features(self, url):
        """Extract features from a single URL"""
        # Basic URL features
        url_length = len(url)
        has_https = 1 if url.startswith('https://') else 0
        num_dots = url.count('.')
        has_at = 1 if '@' in url else 0
        has_dash = 1 if '-' in url else 0
        has_ip = 1 if self._contains_ip(url) else 0
        
        # Additional features for neural network
        num_special_chars = sum(not c.isalnum() and not c.isspace() for c in url)
        num_digits = sum(c.isdigit() for c in url)
        has_underscore = 1 if '_' in url else 0
        has_redirect = 1 if '//' in url.split('/', 3)[-1] else 0
        
        # Normalize features
        features = [
            url_length / 1000.0,  # Normalize URL length
            has_https,
            num_dots / 10.0,      # Normalize dot count
            has_at,
            has_dash,
            has_ip,
            num_special_chars / 50.0,  # Normalize special chars
            num_digits / 20.0,         # Normalize digits
            has_underscore,
            has_redirect
        ]
        
        return features
    
    def _contains_ip(self, url):
        """Check if URL contains IP address"""
        import re
        return bool(re.search(r'(\d{1,3}\.){3}\d{1,3}', url))
    
    def _initialize_model(self):
        """Initialize the neural network model"""
        input_size = len(self._extract_single_url_features("https://example.com"))
        self.model = PhishingNeuralNetwork(input_size=input_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()
        
    def _setup_privacy_engine(self):
        """Setup differential privacy engine"""
        self.privacy_engine = PrivacyEngine()
        
        # Attach privacy engine to model
        self.privacy_engine.attach(
            optimizer=self.optimizer,
            sample_rate=32 / len(self.data),  # Batch size / dataset size
            max_grad_norm=1.0,
            noise_multiplier=self.privacy_config.noise_multiplier
        )
        
    def get_parameters(self, config):
        """Get model parameters for federated aggregation"""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    def set_parameters(self, parameters):
        """Set model parameters from federated aggregation"""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict)
    
    def fit(self, parameters, config):
        """Train the model with differential privacy"""
        self.set_parameters(parameters)
        
        # Training loop with privacy guarantees
        self.model.train()
        total_loss = 0
        
        with BatchMemoryManager(
            data_loader=self.train_loader,
            max_physical_batch_size=16,
            optimizer=self.optimizer
        ) as memory_safe_data_loader:
            
            for epoch in range(5):  # Train for 5 epochs
                epoch_loss = 0
                for batch_idx, (data, target) in enumerate(memory_safe_data_loader):
                    self.optimizer.zero_grad()
                    output = self.model(data)
                    loss = self.criterion(output, target)
                    loss.backward()
                    self.optimizer.step()
                    epoch_loss += loss.item()
                
                total_loss += epoch_loss
                self.epochs_trained += 1
        
        # Update privacy consumption
        self.privacy_consumed += self.privacy_engine.get_epsilon(self.privacy_config.delta)
        
        return self.get_parameters(config), len(self.data), {
            'client_id': self.client_id,
            'privacy_consumed': self.privacy_consumed,
            'epochs_trained': self.epochs_trained,
            'training_loss': total_loss / 5
        }
    
    def evaluate(self, parameters, config):
        """Evaluate the model"""
        self.set_parameters(parameters)
        
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in self.train_loader:
                output = self.model(data)
                total_loss += self.criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        accuracy = correct / total
        avg_loss = total_loss / len(self.train_loader)
        
        return avg_loss, len(self.data), {
            'accuracy': accuracy,
            'client_id': self.client_id,
            'privacy_consumed': self.privacy_consumed
        }

class SecureAggregation:
    """Secure aggregation for federated learning"""
    
    def __init__(self, secret_key: bytes = None):
        self.secret_key = secret_key or Fernet.generate_key()
        self.cipher = Fernet(self.secret_key)
        
    def encrypt_model_update(self, model_update: List[np.ndarray]) -> bytes:
        """Encrypt model update for secure transmission"""
        # Serialize and encrypt model update
        serialized = json.dumps([arr.tolist() for arr in model_update])
        encrypted = self.cipher.encrypt(serialized.encode())
        return encrypted
    
    def decrypt_model_update(self, encrypted_update: bytes) -> List[np.ndarray]:
        """Decrypt model update"""
        decrypted = self.cipher.decrypt(encrypted_update)
        serialized = decrypted.decode()
        update_list = json.loads(serialized)
        return [np.array(arr) for arr in update_list]
    
    def add_verification_hash(self, model_update: bytes, client_id: str) -> Tuple[bytes, str]:
        """Add HMAC verification hash to model update"""
        h = hmac.new(self.secret_key, model_update, hashlib.sha256)
        verification_hash = h.hexdigest()
        return model_update, verification_hash
    
    def verify_update(self, model_update: bytes, verification_hash: str, client_id: str) -> bool:
        """Verify model update integrity"""
        h = hmac.new(self.secret_key, model_update, hashlib.sha256)
        expected_hash = h.hexdigest()
        return hmac.compare_digest(verification_hash, expected_hash)

class FederatedPhishingServer:
    """Federated learning server for coordinating training"""
    
    def __init__(self, privacy_config: DifferentialPrivacyConfig):
        self.privacy_config = privacy_config
        self.secure_agg = SecureAggregation()
        self.clients = {}
        self.global_model = None
        self.training_history = []
        
    def add_client(self, client_id: str, client: FederatedPhishingClient):
        """Add a client to the federated learning system"""
        self.clients[client_id] = client
        
    def start_federated_training(self, num_rounds: int = 10):
        """Start federated training rounds"""
        print(f"Starting federated training with {len(self.clients)} clients for {num_rounds} rounds")
        
        for round_num in range(num_rounds):
            print(f"\n--- Round {round_num + 1} ---")
            
            # Collect model updates from clients
            client_updates = {}
            for client_id, client in self.clients.items():
                try:
                    # Get current global parameters
                    global_params = self.global_model.get_parameters({}) if self.global_model else None
                    
                    if global_params:
                        # Train client model
                        updated_params, num_examples, metrics = client.fit(global_params, {})
                        
                        # Encrypt and verify update
                        encrypted_update = self.secure_agg.encrypt_model_update(updated_params)
                        encrypted_update, verification_hash = self.secure_agg.add_verification_hash(
                            encrypted_update, client_id
                        )
                        
                        client_updates[client_id] = {
                            'encrypted_update': encrypted_update,
                            'verification_hash': verification_hash,
                            'num_examples': num_examples,
                            'metrics': metrics
                        }
                        
                        print(f"Client {client_id}: {metrics['training_loss']:.4f} loss, "
                              f"Privacy consumed: {metrics['privacy_consumed']:.2f}")
                        
                except Exception as e:
                    print(f"Error training client {client_id}: {e}")
            
            # Aggregate model updates securely
            if client_updates:
                self._aggregate_updates(client_updates, round_num)
                
                # Evaluate global model
                self._evaluate_global_model(round_num)
        
        print("\n--- Federated Training Complete ---")
        return self.training_history
    
    def _aggregate_updates(self, client_updates: Dict, round_num: int):
        """Securely aggregate client model updates"""
        # Decrypt and verify all updates
        verified_updates = {}
        total_examples = 0
        
        for client_id, update_info in client_updates.items():
            try:
                # Verify update integrity
                if self.secure_agg.verify_update(
                    update_info['encrypted_update'],
                    update_info['verification_hash'],
                    client_id
                ):
                    # Decrypt update
                    decrypted_update = self.secure_agg.decrypt_model_update(
                        update_info['encrypted_update']
                    )
                    verified_updates[client_id] = {
                        'parameters': decrypted_update,
                        'num_examples': update_info['num_examples'],
                        'metrics': update_info['metrics']
                    }
                    total_examples += update_info['num_examples']
                else:
                    print(f"⚠️ Update verification failed for client {client_id}")
                    
            except Exception as e:
                print(f"Error processing update from client {client_id}: {e}")
        
        if not verified_updates:
            print("❌ No valid updates to aggregate")
            return
        
        # Weighted average aggregation
        aggregated_params = None
        for client_id, update_info in verified_updates.items():
            weight = update_info['num_examples'] / total_examples
            
            if aggregated_params is None:
                aggregated_params = [param * weight for param in update_info['parameters']]
            else:
                for i, param in enumerate(update_info['parameters']):
                    aggregated_params[i] += param * weight
        
        # Update global model
        if self.global_model:
            self.global_model.set_parameters(aggregated_params)
        else:
            # Create new global model if none exists
            self.global_model = self._create_global_model()
            self.global_model.set_parameters(aggregated_params)
        
        # Record round statistics
        round_stats = {
            'round': round_num,
            'clients_participated': len(verified_updates),
            'total_examples': total_examples,
            'avg_privacy_consumed': np.mean([
                update_info['metrics']['privacy_consumed'] 
                for update_info in verified_updates.values()
            ])
        }
        self.training_history.append(round_stats)
        
        print(f"✅ Aggregated updates from {len(verified_updates)} clients")
        print(f"   Total examples: {total_examples}")
        print(f"   Avg privacy consumed: {round_stats['avg_privacy_consumed']:.2f}")
    
    def _create_global_model(self):
        """Create a global model instance"""
        # Use the same architecture as clients
        input_size = 10  # Based on feature extraction
        return PhishingNeuralNetwork(input_size=input_size)
    
    def _evaluate_global_model(self, round_num: int):
        """Evaluate global model performance"""
        if not self.global_model:
            return
        
        # Evaluate on a subset of client data (without exposing raw data)
        total_accuracy = 0
        total_clients = 0
        
        for client_id, client in self.clients.items():
            try:
                global_params = self.global_model.get_parameters({})
                loss, num_examples, metrics = client.evaluate(global_params, {})
                total_accuracy += metrics['accuracy']
                total_clients += 1
            except Exception as e:
                print(f"Error evaluating client {client_id}: {e}")
        
        if total_clients > 0:
            avg_accuracy = total_accuracy / total_clients
            print(f"   Global model accuracy: {avg_accuracy:.4f}")
            
            # Update training history
            self.training_history[-1]['global_accuracy'] = avg_accuracy
    
    def get_privacy_report(self) -> Dict:
        """Generate privacy consumption report"""
        if not self.training_history:
            return {}
        
        total_privacy = sum(round_stats['avg_privacy_consumed'] for round_stats in self.training_history)
        max_privacy = max(round_stats['avg_privacy_consumed'] for round_stats in self.training_history)
        
        return {
            'total_privacy_consumed': total_privacy,
            'max_privacy_per_round': max_privacy,
            'total_rounds': len(self.training_history),
            'privacy_budget_remaining': self.privacy_config.epsilon - total_privacy,
            'privacy_config': self.privacy_config.to_dict()
        }

# Utility functions for data splitting and simulation
def split_data_for_federated_learning(data: pd.DataFrame, num_clients: int, 
                                    client_size_range: Tuple[int, int] = (100, 1000)) -> Dict[str, pd.DataFrame]:
    """Split dataset for federated learning simulation"""
    np.random.seed(42)  # For reproducibility
    
    client_data = {}
    remaining_data = data.copy()
    
    for i in range(num_clients):
        client_id = f"client_{i+1}"
        
        # Randomly select client dataset size
        min_size, max_size = client_size_range
        client_size = np.random.randint(min_size, max_size + 1)
        
        # Ensure we don't exceed remaining data
        client_size = min(client_size, len(remaining_data))
        
        if client_size == 0:
            break
        
        # Randomly sample data for this client
        client_indices = np.random.choice(remaining_data.index, size=client_size, replace=False)
        client_data[client_id] = remaining_data.loc[client_indices].copy()
        
        # Remove selected data from remaining data
        remaining_data = remaining_data.drop(client_indices)
        
        if len(remaining_data) == 0:
            break
    
    return client_data

def simulate_federated_training(data: pd.DataFrame, num_clients: int = 5, 
                              num_rounds: int = 10, privacy_epsilon: float = 1.0):
    """Simulate federated learning training"""
    
    # Split data for clients
    client_data = split_data_for_federated_learning(data, num_clients)
    print(f"Split data among {len(client_data)} clients")
    
    # Setup privacy configuration
    privacy_config = DifferentialPrivacyConfig(epsilon=privacy_epsilon)
    
    # Create federated server
    server = FederatedPhishingServer(privacy_config)
    
    # Create and add clients
    for client_id, client_data_subset in client_data.items():
        client = FederatedPhishingClient(
            client_id=client_id,
            data=client_data_subset,
            privacy_config=privacy_config
        )
        server.add_client(client_id, client)
    
    # Start federated training
    training_history = server.start_federated_training(num_rounds)
    
    # Generate privacy report
    privacy_report = server.get_privacy_report()
    
    return server, training_history, privacy_report

if __name__ == "__main__":
    # Example usage
    print("Federated Learning with Differential Privacy for Phishing Detection")
    print("=" * 70)
    
    # Load sample data (replace with actual data loading)
    # data = pd.read_csv('phishing_dataset.csv')
    
    # Simulate federated training
    # server, history, privacy_report = simulate_federated_training(data)
    
    # print("\nPrivacy Report:")
    # for key, value in privacy_report.items():
    #     print(f"{key}: {value}")
