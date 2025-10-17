"""
Inverse Problem Test Script
===========================
Demonstrates solving the inverse problem: given observation data u_obs,
learn the unknown parameters (a1, a2) using physics-informed neural networks.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from src.model import PINN
from src.train import InverseTrainer
from src.dataloader import load_helmholtz_dataset
from src.visualization import plot_inverse_results, plot_parameter_space_exploration


def run_inverse_experiment():
    """Run inverse problem experiment with different configurations."""
    print("=" * 60)
    print("INVERSE PROBLEM EXPERIMENT")
    print("=" * 60)
    
    # Load dataset
    print("Loading dataset...")
    dataset = load_helmholtz_dataset()
    
    # Select a sample
    sample_idx = 5  # Different sample than forward problem
    sample = dataset.get_sample(sample_idx)
    true_params = sample['params']
    
    print(f"\nSelected sample {sample_idx}:")
    print(f"  True parameters: a1={true_params['a1']:.3f}, a2={true_params['a2']:.3f}, k={true_params['k']:.3f}")
    print(f"  Grid shape: {sample['grid_shape']}")
    
    # Experiment 1: Effect of number of observations
    print("\n" + "="*50)
    print("EXPERIMENT 1: Effect of Number of Observations")
    print("="*50)
    
    n_obs_list = [50, 100, 200, 400]
    results_n_obs = {}
    
    for n_obs in n_obs_list:
        print(f"\nTraining with {n_obs} observations...")
        
        model = PINN([64, 64, 64, 1], activation="tanh", pos_enc=24)
        trainer = InverseTrainer(model, learning_rate=1e-3, param_lr=1e-1)
        
        results = trainer.train(
            dataset, sample_idx,
            n_epochs=2000,
            n_collocation=2000,
            n_obs=n_obs,
            data_weight=10.0,
            pde_weight=1.0,
            noise_level=0.01,
            print_every=400
        )
        
        results_n_obs[n_obs] = results
        print(f"Final errors with {n_obs} obs: a1={results['param_errors']['a1']:.1%}, a2={results['param_errors']['a2']:.1%}")
    
    # Plot observation number comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Parameter errors vs number of observations
    a1_errors = [results_n_obs[n]['param_errors']['a1'] * 100 for n in n_obs_list]
    a2_errors = [results_n_obs[n]['param_errors']['a2'] * 100 for n in n_obs_list]
    
    axes[0].plot(n_obs_list, a1_errors, 'bo-', linewidth=2, markersize=8, label='a1 error')
    axes[0].plot(n_obs_list, a2_errors, 'ro-', linewidth=2, markersize=8, label='a2 error')
    axes[0].set_xlabel('Number of Observations')
    axes[0].set_ylabel('Parameter Error (%)')
    axes[0].set_title('Parameter Error vs Data Amount')
    axes[0].legend()
    axes[0].grid(True)
    axes[0].set_yscale('log')
    
    # Final losses
    final_losses = [results_n_obs[n]['loss_history'][-1]['total_loss'] for n in n_obs_list]
    axes[1].semilogy(n_obs_list, final_losses, 'go-', linewidth=2, markersize=8)
    axes[1].set_xlabel('Number of Observations')
    axes[1].set_ylabel('Final Loss')
    axes[1].set_title('Convergence vs Data Amount')
    axes[1].grid(True)
    
    # Parameter trajectories for different data amounts
    for i, n_obs in enumerate([50, 400]):
        epochs = range(len(results_n_obs[n_obs]['param_history']))
        a1_traj = [p[0] for p in results_n_obs[n_obs]['param_history']]
        color = ['red', 'blue'][i]
        axes[2].plot(epochs, a1_traj, color=color, linewidth=2, label=f'{n_obs} obs')
    
    axes[2].axhline(true_params['a1'], color='black', linestyle='--', linewidth=2, label='True a1')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('a1 Prediction')
    axes[2].set_title('Parameter Learning Speed')
    axes[2].legend()
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig('inverse_observations_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Experiment 2: Effect of noise level
    print("\n" + "="*50)
    print("EXPERIMENT 2: Effect of Observation Noise")
    print("="*50)
    
    noise_levels = [0.0, 0.01, 0.03, 0.05]
    results_noise = {}
    
    for noise in noise_levels:
        print(f"\nTraining with {noise:.1%} noise...")
        
        model = PINN([64, 64, 64, 1], activation="tanh", pos_enc=2)
        trainer = InverseTrainer(model, learning_rate=1e-3, param_lr=1e-2)
        
        results = trainer.train(
            dataset, sample_idx,
            n_epochs=2000,
            n_collocation=2000,
            n_obs=200,
            data_weight=10.0,
            pde_weight=1.0,
            noise_level=noise,
            print_every=400
        )
        
        results_noise[noise] = results
        print(f"Final errors with {noise:.1%} noise: a1={results['param_errors']['a1']:.1%}, a2={results['param_errors']['a2']:.1%}")
    
    # Plot noise comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    noise_percentages = [n * 100 for n in noise_levels]
    a1_errors_noise = [results_noise[n]['param_errors']['a1'] * 100 for n in noise_levels]
    a2_errors_noise = [results_noise[n]['param_errors']['a2'] * 100 for n in noise_levels]
    
    axes[0].plot(noise_percentages, a1_errors_noise, 'bo-', linewidth=2, markersize=8, label='a1 error')
    axes[0].plot(noise_percentages, a2_errors_noise, 'ro-', linewidth=2, markersize=8, label='a2 error')
    axes[0].set_xlabel('Noise Level (%)')
    axes[0].set_ylabel('Parameter Error (%)')
    axes[0].set_title('Parameter Error vs Noise Level')
    axes[0].legend()
    axes[0].grid(True)
    
    # Loss evolution comparison
    for noise in [0.0, 0.05]:
        epochs = range(len(results_noise[noise]['loss_history']))
        losses = [h['total_loss'] for h in results_noise[noise]['loss_history']]
        label = f'{noise:.1%} noise' if noise > 0 else 'No noise'
        axes[1].semilogy(epochs, losses, linewidth=2, label=label)
    
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Total Loss')
    axes[1].set_title('Loss Evolution: Clean vs Noisy Data')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('inverse_noise_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Experiment 3: Detailed analysis of best configuration
    print("\n" + "="*50)
    print("EXPERIMENT 3: Detailed Analysis")
    print("="*50)
    
    print("\nTraining with optimal configuration...")
    model_best = PINN([64, 64, 64, 1], activation="tanh", pos_enc=4)
    trainer_best = InverseTrainer(model_best, learning_rate=1e-3, param_lr=5e-3)
    
    results_best = trainer_best.train(
        dataset, sample_idx,
        n_epochs=3000,
        n_collocation=2500,
        n_obs=300,
        data_weight=15.0,
        pde_weight=1.0,
        noise_level=0.02,
        print_every=500
    )
    
    # Detailed plotting
    plot_inverse_results(results_best, 'inverse_detailed_results.png')
    
    # Summary
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    print("\nNumber of Observations Results:")
    for n_obs, results in results_n_obs.items():
        print(f"  {n_obs:>3} obs: a1 error = {results['param_errors']['a1']:.1%}, a2 error = {results['param_errors']['a2']:.1%}")
    
    print("\nNoise Level Results:")
    for noise, results in results_noise.items():
        print(f"  {noise:>4.1%} noise: a1 error = {results['param_errors']['a1']:.1%}, a2 error = {results['param_errors']['a2']:.1%}")
    
    print(f"\nBest configuration:")
    print(f"  True: a1={results_best['true_params']['a1']:.3f}, a2={results_best['true_params']['a2']:.3f}")
    print(f"  Pred: a1={results_best['final_params']['a1']:.3f}, a2={results_best['final_params']['a2']:.3f}")
    print(f"  Errors: a1={results_best['param_errors']['a1']:.1%}, a2={results_best['param_errors']['a2']:.1%}")
    
    return {
        'observations': results_n_obs,
        'noise': results_noise,
        'best_result': results_best
    }


if __name__ == "__main__":
    # Run the inverse problem experiment
    torch.manual_seed(42)  # For reproducibility
    results = run_inverse_experiment()
    
    print("\nInverse problem experiment completed!")
    print("Generated plots:")
    print("  - inverse_observations_comparison.png")
    print("  - inverse_noise_comparison.png")
    print("  - inverse_detailed_results.png")