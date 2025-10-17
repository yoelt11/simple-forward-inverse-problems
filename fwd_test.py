"""
Forward Problem Test Script
===========================
Demonstrates solving the forward problem: given known parameters (a1, a2, k),
learn the solution u(x,y) using physics-informed neural networks.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from src.model import PINN
from src.train import ForwardTrainer
from src.dataloader import load_helmholtz_dataset
from src.visualization import plot_forward_results, plot_solution_comparison


def run_forward_experiment():
    """Run forward problem experiment with different configurations."""
    print("=" * 60)
    print("FORWARD PROBLEM EXPERIMENT")
    print("=" * 60)
    
    # Load dataset
    print("Loading dataset...")
    dataset = load_helmholtz_dataset()
    
    # Select a sample
    sample_idx = 0
    sample = dataset.get_sample(sample_idx)
    params = sample['params']
    
    print(f"\nSelected sample {sample_idx}:")
    print(f"  Parameters: a1={params['a1']:.3f}, a2={params['a2']:.3f}, k={params['k']:.3f}")
    print(f"  Grid shape: {sample['grid_shape']}")
    
    # Experiment 1: Different activation functions
    print("\n" + "="*50)
    print("EXPERIMENT 1: Effect of Activation Functions")
    print("="*50)
    
    activations = ["tanh", "relu", "silu", "gelu"]
    results_activations = {}
    
    for activation in activations:
        print(f"\nTraining with {activation} activation...")
        
        model = PINN([64, 64, 64, 1], activation=activation, pos_enc=0)
        trainer = ForwardTrainer(model, learning_rate=1e-3)
        
        results = trainer.train(
            dataset, sample_idx, 
            n_epochs=2000, 
            n_collocation=2500,
            bc_weight=1.0, 
            pde_weight=1.0,
            print_every=500
        )
        
        results_activations[activation] = results
        print(f"Final L2 error with {activation}: {results['final_l2_error']:.2e}")
    
    # Plot activation comparison
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    
    for i, (activation, results) in enumerate(results_activations.items()):
        axes[i].imshow(results['u_pred'], extent=[-1, 1, -1, 1], 
                      origin='lower', cmap='RdBu_r')
        axes[i].set_title(f'{activation}: L2={results["final_l2_error"]:.2e}')
        axes[i].set_xlabel('x')
        axes[i].set_ylabel('y')
    
    plt.tight_layout()
    plt.savefig('forward_activations_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Experiment 2: Effect of positional encoding
    print("\n" + "="*50)
    print("EXPERIMENT 2: Effect of Positional Encoding")
    print("="*50)
    
    pos_encodings = [0, 2, 4, 6]
    results_pe = {}
    
    for pe in pos_encodings:
        print(f"\nTraining with positional encoding = {pe}...")
        
        model = PINN([64, 64, 64, 1], activation="tanh", pos_enc=pe)
        trainer = ForwardTrainer(model, learning_rate=1e-3)
        
        results = trainer.train(
            dataset, sample_idx,
            n_epochs=2000,
            n_collocation=2500,
            bc_weight=1.0,
            pde_weight=1.0,
            print_every=500
        )
        
        results_pe[pe] = results
        print(f"Final L2 error with PE={pe}: {results['final_l2_error']:.2e}")
    
    # Plot PE comparison
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    
    for i, (pe, results) in enumerate(results_pe.items()):
        axes[i].imshow(results['u_pred'], extent=[-1, 1, -1, 1], 
                      origin='lower', cmap='RdBu_r')
        axes[i].set_title(f'PE={pe}: L2={results["final_l2_error"]:.2e}')
        axes[i].set_xlabel('x')
        axes[i].set_ylabel('y')
    
    plt.tight_layout()
    plt.savefig('forward_positional_encoding_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Experiment 3: Detailed analysis of best configuration
    print("\n" + "="*50)
    print("EXPERIMENT 3: Detailed Analysis")
    print("="*50)
    
    print("\nTraining with best configuration (tanh + PE=4)...")
    model_best = PINN([64, 64, 64, 1], activation="tanh", pos_enc=4)
    trainer_best = ForwardTrainer(model_best, learning_rate=1e-3)
    
    results_best = trainer_best.train(
        dataset, sample_idx,
        n_epochs=3000,
        n_collocation=3000,
        bc_weight=1.0,
        pde_weight=1.0,
        print_every=500
    )
    
    # Detailed plotting
    plot_forward_results(results_best, sample, 'forward_detailed_results.png')
    
    # Summary
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    print("\nActivation Function Results:")
    for activation, results in results_activations.items():
        print(f"  {activation:>6}: L2 error = {results['final_l2_error']:.2e}")
    
    print("\nPositional Encoding Results:")
    for pe, results in results_pe.items():
        print(f"  PE={pe:>2}: L2 error = {results['final_l2_error']:.2e}")
    
    print(f"\nBest configuration: L2 error = {results_best['final_l2_error']:.2e}")
    
    return {
        'activations': results_activations,
        'positional_encoding': results_pe,
        'best_result': results_best,
        'sample_data': sample
    }


if __name__ == "__main__":
    # Run the forward problem experiment
    torch.manual_seed(42)  # For reproducibility
    results = run_forward_experiment()
    
    print("\nForward problem experiment completed!")
    print("Generated plots:")
    print("  - forward_activations_comparison.png")
    print("  - forward_positional_encoding_comparison.png") 
    print("  - forward_detailed_results.png")