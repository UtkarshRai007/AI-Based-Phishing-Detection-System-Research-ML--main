#!/usr/bin/env python3
"""
Demonstration script for Federated Learning with Differential Privacy
for Phishing Detection

This script demonstrates how to:
1. Set up federated learning clients and server
2. Train models with differential privacy
3. Monitor privacy consumption
4. Evaluate model performance
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import time
import json

# Import our federated learning modules
from federated_learning import (
    FederatedPhishingServer,
    FederatedPhishingClient,
    DifferentialPrivacyConfig,
    simulate_federated_training,
    split_data_for_federated_learning
)
from federated_config import (
    FederatedConfig,
    PrivacyConfig,
    get_preset_config,
    calculate_privacy_budget,
    validate_privacy_parameters
)

def load_and_prepare_data(filepath: str = 'phishing_dataset.csv') -> pd.DataFrame:
    """Load and prepare the phishing dataset"""
    print(f"📊 Loading dataset from {filepath}...")
    
    try:
        df = pd.read_csv(filepath)
        print(f"   Original dataset shape: {df.shape}")
        
        # Clean and prepare data
        df.columns = df.columns.str.strip().str.lower()
        df['label'] = df['label'].astype(str).str.strip().str.lower()
        df['label'] = df['label'].fillna('legitimate')
        df['label'] = df['label'].map({'bad': 1, 'good': 0, 'legitimate': 0})
        df = df.dropna(subset=['label'])
        
        print(f"   Cleaned dataset shape: {df.shape}")
        print(f"   Label distribution:")
        print(f"     - Phishing (1): {df['label'].sum()}")
        print(f"     - Legitimate (0): {(df['label'] == 0).sum()}")
        
        return df
        
    except FileNotFoundError:
        print(f"❌ Error: Dataset file '{filepath}' not found!")
        print("   Please ensure the phishing_dataset.csv file is in the current directory.")
        return None
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return None

def demonstrate_privacy_configurations():
    """Demonstrate different privacy configurations"""
    print("\n🔒 Differential Privacy Configurations")
    print("=" * 50)
    
    # Test different privacy parameters
    test_configs = [
        ("High Privacy", 0.5, 1e-6),
        ("Balanced", 1.0, 1e-5),
        ("High Accuracy", 2.0, 1e-4),
        ("Fast Training", 1.5, 1e-4)
    ]
    
    for name, epsilon, delta in test_configs:
        is_valid, message = validate_privacy_parameters(epsilon, delta)
        status = "✅" if is_valid else "❌"
        print(f"{status} {name}: ε={epsilon}, δ={delta:.0e} - {message}")
        
        if is_valid:
            budget = calculate_privacy_budget(epsilon, delta, 10)
            print(f"   Privacy budget for 10 rounds: ε={budget['total_epsilon']:.2f}, δ={budget['total_delta']:.0e}")

def demonstrate_federated_training(data: pd.DataFrame, 
                                num_clients: int = 5,
                                num_rounds: int = 10,
                                privacy_epsilon: float = 1.0):
    """Demonstrate federated learning training"""
    print(f"\n🤝 Federated Learning Training")
    print("=" * 50)
    print(f"Configuration:")
    print(f"   - Number of clients: {num_clients}")
    print(f"   - Training rounds: {num_rounds}")
    print(f"   - Privacy budget (ε): {privacy_epsilon}")
    
    # Split data for clients
    print(f"\n📊 Splitting data among {num_clients} clients...")
    client_data = split_data_for_federated_learning(data, num_clients)
    
    print(f"   Data distribution:")
    for client_id, client_data_subset in client_data.items():
        print(f"     - {client_id}: {len(client_data_subset)} samples")
    
    # Setup privacy configuration
    privacy_config = DifferentialPrivacyConfig(epsilon=privacy_epsilon)
    
    # Create federated server
    print(f"\n🚀 Creating federated learning server...")
    server = FederatedPhishingServer(privacy_config)
    
    # Create and add clients
    print(f"   Creating {len(client_data)} clients...")
    for client_id, client_data_subset in client_data.items():
        client = FederatedPhishingClient(
            client_id=client_id,
            data=client_data_subset,
            privacy_config=privacy_config
        )
        server.add_client(client_id, client)
        print(f"     ✅ {client_id} created with {len(client_data_subset)} samples")
    
    # Start federated training
    print(f"\n🔄 Starting federated training for {num_rounds} rounds...")
    start_time = time.time()
    
    training_history = server.start_federated_training(num_rounds)
    
    training_time = time.time() - start_time
    print(f"\n✅ Federated training completed in {training_time:.2f} seconds")
    
    # Generate privacy report
    privacy_report = server.get_privacy_report()
    
    return server, training_history, privacy_report, training_time

def analyze_training_results(training_history: List[Dict], privacy_report: Dict):
    """Analyze and visualize training results"""
    print(f"\n📊 Training Results Analysis")
    print("=" * 50)
    
    if not training_history:
        print("❌ No training history available")
        return
    
    # Extract metrics
    rounds = [r['round'] for r in training_history]
    accuracies = [r.get('global_accuracy', 0) for r in training_history]
    clients_participated = [r['clients_participated'] for r in training_history]
    total_examples = [r['total_examples'] for r in training_history]
    privacy_consumed = [r['avg_privacy_consumed'] for r in training_history]
    
    # Performance metrics
    final_accuracy = accuracies[-1] if accuracies else 0
    best_accuracy = max(accuracies) if accuracies else 0
    avg_clients = np.mean(clients_participated)
    total_examples_processed = sum(total_examples)
    
    print(f"Model Performance:")
    print(f"   - Final accuracy: {final_accuracy:.4f}")
    print(f"   - Best accuracy: {best_accuracy:.4f}")
    print(f"   - Average clients per round: {avg_clients:.1f}")
    print(f"   - Total examples processed: {total_examples_processed:,}")
    
    # Privacy metrics
    if privacy_report:
        print(f"\nPrivacy Analysis:")
        print(f"   - Total privacy consumed: {privacy_report['total_privacy_consumed']:.3f} ε")
        print(f"   - Privacy budget remaining: {privacy_report['privacy_budget_remaining']:.3f} ε")
        print(f"   - Max privacy per round: {privacy_report['max_privacy_per_round']:.3f} ε")
        print(f"   - Privacy efficiency: {total_examples_processed / privacy_report['total_privacy_consumed']:.0f} examples/ε")
    
    # Create visualizations
    create_training_visualizations(rounds, accuracies, clients_participated, 
                                 total_examples, privacy_consumed)

def create_training_visualizations(rounds, accuracies, clients_participated, 
                                 total_examples, privacy_consumed):
    """Create training visualization plots"""
    print(f"\n📈 Creating training visualizations...")
    
    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Federated Learning Training Results', fontsize=16, fontweight='bold')
    
    # 1. Accuracy over rounds
    axes[0, 0].plot(rounds, accuracies, 'b-o', linewidth=2, markersize=6)
    axes[0, 0].set_title('Model Accuracy Over Training Rounds')
    axes[0, 0].set_xlabel('Training Round')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim(0, 1)
    
    # 2. Clients participation
    axes[0, 1].bar(rounds, clients_participated, color='green', alpha=0.7)
    axes[0, 1].set_title('Clients Participating per Round')
    axes[0, 1].set_xlabel('Training Round')
    axes[0, 1].set_ylabel('Number of Clients')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Examples processed
    axes[1, 0].plot(rounds, total_examples, 'r-s', linewidth=2, markersize=6)
    axes[1, 0].set_title('Examples Processed per Round')
    axes[1, 0].set_xlabel('Training Round')
    axes[1, 0].set_ylabel('Number of Examples')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Privacy consumption
    axes[1, 1].plot(rounds, privacy_consumed, 'purple', marker='^', linewidth=2, markersize=6)
    axes[1, 1].set_title('Privacy Consumption per Round')
    axes[1, 1].set_xlabel('Training Round')
    axes[1, 1].set_ylabel('Privacy Consumed (ε)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    plot_filename = 'federated_training_results.png'
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"   📊 Training visualizations saved as '{plot_filename}'")
    
    # Show the plot
    plt.show()

def demonstrate_preset_configurations(data: pd.DataFrame):
    """Demonstrate different preset configurations"""
    print(f"\n⚙️ Preset Configuration Comparison")
    print("=" * 50)
    
    preset_names = ['high_privacy', 'balanced', 'high_accuracy', 'fast_training']
    results = {}
    
    for preset_name in preset_names:
        print(f"\n🔄 Testing {preset_name} preset...")
        
        try:
            config = get_preset_config(preset_name)
            
            # Run federated training with this preset
            server, history, privacy_report, training_time = demonstrate_federated_training(
                data=data,
                num_clients=config.num_clients,
                num_rounds=config.num_rounds,
                privacy_epsilon=config.epsilon
            )
            
            # Store results
            results[preset_name] = {
                'config': config,
                'final_accuracy': history[-1].get('global_accuracy', 0) if history else 0,
                'training_time': training_time,
                'privacy_consumed': privacy_report['total_privacy_consumed'] if privacy_report else 0,
                'rounds': len(history)
            }
            
            print(f"   ✅ {preset_name} completed successfully")
            
        except Exception as e:
            print(f"   ❌ Error with {preset_name}: {e}")
            results[preset_name] = None
    
    # Compare results
    print(f"\n📊 Preset Configuration Comparison")
    print("=" * 50)
    
    comparison_data = []
    for preset_name, result in results.items():
        if result:
            comparison_data.append({
                'Preset': preset_name,
                'Accuracy': f"{result['final_accuracy']:.4f}",
                'Training Time (s)': f"{result['training_time']:.2f}",
                'Privacy Used (ε)': f"{result['privacy_consumed']:.3f}",
                'Rounds': result['rounds']
            })
    
    if comparison_data:
        comparison_df = pd.DataFrame(comparison_data)
        print(comparison_df.to_string(index=False))
        
        # Save comparison results
        with open('preset_comparison.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n💾 Comparison results saved to 'preset_comparison.json'")

def main():
    """Main demonstration function"""
    print("🛡️ Federated Learning with Differential Privacy for Phishing Detection")
    print("=" * 80)
    print("This demonstration shows how to use federated learning with differential privacy")
    print("to train phishing detection models while preserving data privacy.")
    print()
    
    # Load dataset
    data = load_and_prepare_data()
    if data is None:
        print("❌ Cannot proceed without dataset. Exiting.")
        return
    
    # Demonstrate privacy configurations
    demonstrate_privacy_configurations()
    
    # Demonstrate federated training
    print(f"\n" + "="*80)
    print("🚀 MAIN DEMONSTRATION: Federated Learning Training")
    print("="*80)
    
    server, training_history, privacy_report, training_time = demonstrate_federated_training(
        data=data,
        num_clients=5,
        num_rounds=10,
        privacy_epsilon=1.0
    )
    
    # Analyze results
    analyze_training_results(training_history, privacy_report)
    
    # Demonstrate preset configurations
    print(f"\n" + "="*80)
    print("⚙️ ADVANCED DEMONSTRATION: Preset Configuration Comparison")
    print("="*80)
    
    demonstrate_preset_configurations(data)
    
    # Summary
    print(f"\n" + "="*80)
    print("🎉 DEMONSTRATION COMPLETED SUCCESSFULLY!")
    print("="*80)
    print("What we've demonstrated:")
    print("✅ Federated learning setup with multiple clients")
    print("✅ Differential privacy with configurable parameters")
    print("✅ Secure model aggregation without data sharing")
    print("✅ Privacy budget management and monitoring")
    print("✅ Performance analysis and visualization")
    print("✅ Configuration presets for different use cases")
    print()
    print("Next steps:")
    print("1. Run the enhanced GUI: streamlit run enhanced_phishing_gui.py")
    print("2. Experiment with different privacy parameters")
    print("3. Scale to more clients and rounds")
    print("4. Integrate with real-world deployment scenarios")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Demonstration interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
