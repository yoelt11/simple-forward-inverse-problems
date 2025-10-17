"""
Visualization utilities for PINNs forward and inverse problems.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional, Union


def plot_solution_comparison(u_true: torch.Tensor, u_pred: torch.Tensor, 
                           title_prefix: str = "", domain: tuple = (-1, 1),
                           save_path: Optional[str] = None) -> None:
    """Plot true vs predicted solution with error using contour plots."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Create coordinate grids
    ny, nx = u_true.shape
    x = np.linspace(domain[0], domain[1], nx)
    y = np.linspace(domain[0], domain[1], ny)
    X, Y = np.meshgrid(x, y)
    levels = 100
    
    # True solution
    cs1 = axes[0].contourf(X, Y, u_true.numpy(), levels=levels, cmap='RdBu_r')
    axes[0].contour(X, Y, u_true.numpy(), levels=levels, colors='black', alpha=0.3, linewidths=0.5)
    axes[0].set_title(f'{title_prefix}True Solution')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    plt.colorbar(cs1, ax=axes[0])
    
    # Predicted solution
    cs2 = axes[1].contourf(X, Y, u_pred.detach().numpy(), levels=levels, cmap='RdBu_r')
    axes[1].contour(X, Y, u_pred.detach().numpy(), levels=levels, colors='black', alpha=0.3, linewidths=0.5)
    axes[1].set_title(f'{title_prefix}Predicted Solution')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')
    plt.colorbar(cs2, ax=axes[1])
    
    # Error
    error = torch.abs(u_pred - u_true)
    l2_error = torch.norm(u_pred - u_true) / torch.norm(u_true)
    cs3 = axes[2].contourf(X, Y, error.detach().numpy(), levels=levels, cmap='Reds')
    axes[2].contour(X, Y, error.detach().numpy(), levels=levels, colors='black', alpha=0.3, linewidths=0.5)
    axes[2].set_title(f'{title_prefix}Error (L2: {l2_error:.2e})')
    axes[2].set_xlabel('x')
    axes[2].set_ylabel('y')
    plt.colorbar(cs3, ax=axes[2])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def plot_loss_curves(loss_history: List[Dict], title: str = "Loss Evolution",
                    save_path: Optional[str] = None) -> None:
    """Plot loss evolution curves."""
    epochs = range(len(loss_history))
    
    # Extract all available loss components
    loss_keys = list(loss_history[0].keys())
    n_components = len(loss_keys)
    
    # Create subplots
    if n_components <= 2:
        fig, axes = plt.subplots(1, n_components, figsize=(6*n_components, 4))
        if n_components == 1:
            axes = [axes]
    else:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.flatten()
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    for i, key in enumerate(loss_keys):
        if i >= len(axes):
            break
            
        values = [h[key] for h in loss_history]
        axes[i].semilogy(epochs, values, color=colors[i % len(colors)], linewidth=2)
        axes[i].set_xlabel('Epoch')
        axes[i].set_ylabel('Loss')
        axes[i].set_title(f'{key.replace("_", " ").title()}')
        axes[i].grid(True)
    
    # Hide unused subplots
    for i in range(len(loss_keys), len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def plot_cross_section(u_true: torch.Tensor, u_pred: torch.Tensor, 
                      axis: str = 'x', position: float = 0.0,
                      domain: tuple = (-1, 1), title: str = "",
                      save_path: Optional[str] = None) -> None:
    """Plot cross-section comparison along specified axis."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    
    if axis == 'x':
        # Cross-section along x-axis at fixed y
        idx = int((position - domain[0]) / (domain[1] - domain[0]) * u_true.shape[0])
        idx = max(0, min(idx, u_true.shape[0] - 1))
        
        coords = np.linspace(domain[0], domain[1], u_true.shape[1])
        true_vals = u_true[idx, :]
        pred_vals = u_pred[idx, :]
        xlabel = 'x'
        section_title = f'y = {position:.1f}'
        
    else:  # axis == 'y'
        # Cross-section along y-axis at fixed x
        idx = int((position - domain[0]) / (domain[1] - domain[0]) * u_true.shape[1])
        idx = max(0, min(idx, u_true.shape[1] - 1))
        
        coords = np.linspace(domain[0], domain[1], u_true.shape[0])
        true_vals = u_true[:, idx]
        pred_vals = u_pred[:, idx]
        xlabel = 'y'
        section_title = f'x = {position:.1f}'
    
    ax.plot(coords, true_vals, 'b-', linewidth=2, label='True')
    ax.plot(coords, pred_vals, 'r--', linewidth=2, label='Predicted')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('u')
    ax.set_title(f'{title}Cross-section at {section_title}')
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def plot_forward_results(results: Dict, sample_data: Dict, 
                        save_path: Optional[str] = None) -> None:
    """Comprehensive plotting for forward problem results using contour plots."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    # Create coordinate grids
    ny, nx = sample_data['solution'].shape
    x = np.linspace(-1, 1, nx)
    y = np.linspace(-1, 1, ny)
    X, Y = np.meshgrid(x, y)
    levels = 50
    
    # True solution
    cs1 = axes[0, 0].contourf(X, Y, sample_data['solution'].numpy(), levels=levels, cmap='RdBu_r')
    axes[0, 0].contour(X, Y, sample_data['solution'].numpy(), levels=levels, colors='black', alpha=0.3, linewidths=0.5)
    axes[0, 0].set_title('True Solution')
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('y')
    plt.colorbar(cs1, ax=axes[0, 0])
    
    # Predicted solution
    cs2 = axes[0, 1].contourf(X, Y, results['u_pred'].detach().numpy(), levels=levels, cmap='RdBu_r')
    axes[0, 1].contour(X, Y, results['u_pred'].detach().numpy(), levels=levels, colors='black', alpha=0.3, linewidths=0.5)
    axes[0, 1].set_title('PINN Prediction')
    axes[0, 1].set_xlabel('x')
    axes[0, 1].set_ylabel('y')
    plt.colorbar(cs2, ax=axes[0, 1])
    
    # Error
    error = torch.abs(results['u_pred'] - sample_data['solution'])
    cs3 = axes[0, 2].contourf(X, Y, error.detach().numpy(), levels=levels, cmap='Reds')
    axes[0, 2].contour(X, Y, error.detach().numpy(), levels=levels, colors='black', alpha=0.3, linewidths=0.5)
    axes[0, 2].set_title(f'Absolute Error\nL2: {results["final_l2_error"]:.2e}')
    axes[0, 2].set_xlabel('x')
    axes[0, 2].set_ylabel('y')
    plt.colorbar(cs3, ax=axes[0, 2])
    
    # Loss curves
    epochs = range(len(results['loss_history']))
    total_losses = [h['total_loss'] for h in results['loss_history']]
    pde_losses = [h['pde_loss'] for h in results['loss_history']]
    bc_losses = [h['bc_loss'] for h in results['loss_history']]
    
    axes[1, 0].semilogy(epochs, total_losses, 'b-', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].set_title('Total Loss')
    axes[1, 0].grid(True)
    
    axes[1, 1].semilogy(epochs, pde_losses, 'r-', linewidth=2, label='PDE Loss')
    axes[1, 1].semilogy(epochs, bc_losses, 'g-', linewidth=2, label='BC Loss')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].set_title('Loss Components')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    # Cross-section comparison
    mid_idx = sample_data['solution'].shape[0] // 2
    x_vals = np.linspace(-1, 1, sample_data['solution'].shape[1])
    
    axes[1, 2].plot(x_vals, sample_data['solution'][mid_idx, :], 'b-', 
                   label='True', linewidth=2)
    axes[1, 2].plot(x_vals, results['u_pred'][mid_idx, :], 'r--', 
                   label='PINN', linewidth=2)
    axes[1, 2].set_xlabel('x')
    axes[1, 2].set_ylabel('u(x, y=0)')
    axes[1, 2].set_title('Cross-section at y=0')
    axes[1, 2].legend()
    axes[1, 2].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def plot_inverse_results(results: Dict, save_path: Optional[str] = None) -> None:
    """Comprehensive plotting for inverse problem results."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Parameter evolution
    epochs = range(len(results['param_history']))
    a1_pred = [p[0] for p in results['param_history']]
    a2_pred = [p[1] for p in results['param_history']]
    
    true_a1 = results['true_params']['a1']
    true_a2 = results['true_params']['a2']
    
    axes[0, 0].plot(epochs, a1_pred, 'b-', linewidth=2, label='Predicted a1')
    axes[0, 0].axhline(true_a1, color='r', linestyle='--', linewidth=2, label='True a1')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('a1')
    axes[0, 0].set_title('Parameter a1 Evolution')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    axes[0, 1].plot(epochs, a2_pred, 'g-', linewidth=2, label='Predicted a2')
    axes[0, 1].axhline(true_a2, color='r', linestyle='--', linewidth=2, label='True a2')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('a2')
    axes[0, 1].set_title('Parameter a2 Evolution')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Parameter errors
    a1_errors = [abs(p - true_a1) / true_a1 for p in a1_pred]
    a2_errors = [abs(p - true_a2) / true_a2 for p in a2_pred]
    
    axes[0, 2].semilogy(epochs, a1_errors, 'b-', linewidth=2, label='a1 error')
    axes[0, 2].semilogy(epochs, a2_errors, 'g-', linewidth=2, label='a2 error')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Relative Error')
    axes[0, 2].set_title('Parameter Convergence')
    axes[0, 2].legend()
    axes[0, 2].grid(True)
    
    # Loss evolution
    total_losses = [h['total_loss'] for h in results['loss_history']]
    pde_losses = [h['pde_loss'] for h in results['loss_history']]
    data_losses = [h['data_loss'] for h in results['loss_history']]
    
    axes[1, 0].semilogy(epochs, total_losses, 'k-', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].set_title('Total Loss')
    axes[1, 0].grid(True)
    
    axes[1, 1].semilogy(epochs, pde_losses, 'r-', linewidth=2, label='PDE Loss')
    axes[1, 1].semilogy(epochs, data_losses, 'b-', linewidth=2, label='Data Loss')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].set_title('Loss Components')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    # Final parameter comparison
    params_true = [true_a1, true_a2]
    params_pred = [results['final_params']['a1'], results['final_params']['a2']]
    param_names = ['a1', 'a2']
    
    x_pos = np.arange(len(param_names))
    width = 0.35
    
    axes[1, 2].bar(x_pos - width/2, params_true, width, label='True', alpha=0.8)
    axes[1, 2].bar(x_pos + width/2, params_pred, width, label='Predicted', alpha=0.8)
    axes[1, 2].set_xlabel('Parameters')
    axes[1, 2].set_ylabel('Value')
    axes[1, 2].set_title('Final Parameter Comparison')
    axes[1, 2].set_xticks(x_pos)
    axes[1, 2].set_xticklabels(param_names)
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    # Add text with final errors
    error_text = f"Final Errors:\na1: {results['param_errors']['a1']:.1%}\na2: {results['param_errors']['a2']:.1%}"
    axes[1, 2].text(0.02, 0.98, error_text, transform=axes[1, 2].transAxes, 
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def plot_experiment_comparison(results_dict: Dict, metric_key: str, 
                             x_label: str, title: str,
                             save_path: Optional[str] = None) -> None:
    """Plot comparison across different experimental configurations."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    x_values = list(results_dict.keys())
    y_values = []
    
    for key, results in results_dict.items():
        if isinstance(results, dict) and metric_key in results:
            y_values.append(results[metric_key])
        else:
            # Assume it's a nested structure
            y_values.append(results['final_l2_error'])  # Default fallback
    
    ax.plot(x_values, y_values, 'bo-', linewidth=2, markersize=8)
    ax.set_xlabel(x_label)
    ax.set_ylabel(metric_key.replace('_', ' ').title())
    ax.set_title(title)
    ax.grid(True)
    ax.set_yscale('log')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def plot_parameter_space_exploration(param_history: List, true_params: Dict,
                                   save_path: Optional[str] = None) -> None:
    """Plot parameter space exploration during inverse problem training."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    
    a1_vals = [p[0] for p in param_history]
    a2_vals = [p[1] for p in param_history]
    
    # Plot trajectory
    ax.plot(a1_vals, a2_vals, 'b-', linewidth=2, alpha=0.7, label='Learning trajectory')
    ax.plot(a1_vals[0], a2_vals[0], 'go', markersize=10, label='Initial guess')
    ax.plot(a1_vals[-1], a2_vals[-1], 'bo', markersize=10, label='Final prediction')
    ax.plot(true_params['a1'], true_params['a2'], 'r*', markersize=15, label='True parameters')
    
    ax.set_xlabel('a1')
    ax.set_ylabel('a2')
    ax.set_title('Parameter Space Exploration')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def plot_observation_points(X_obs: torch.Tensor, u_obs: torch.Tensor,
                          domain: tuple = (-1, 1), title: str = "Observation Points",
                          save_path: Optional[str] = None) -> None:
    """Plot observation points used for inverse problem."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    scatter = ax.scatter(X_obs[:, 0], X_obs[:, 1], c=u_obs, cmap='RdBu_r', s=50)
    ax.set_xlim(domain)
    ax.set_ylim(domain)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'{title} (N={len(X_obs)})')
    plt.colorbar(scatter, ax=ax, label='u(x,y)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


# Example usage and testing
if __name__ == "__main__":
    # Create dummy data for testing
    x = torch.linspace(-1, 1, 64)
    y = torch.linspace(-1, 1, 64)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    
    # Test solution
    u_true = torch.sin(2 * np.pi * X) * torch.sin(2 * np.pi * Y)
    u_pred = u_true + 0.1 * torch.randn_like(u_true)
    
    print("Testing visualization functions...")
    
    # Test solution comparison
    plot_solution_comparison(u_true, u_pred, "Test: ")
    
    # Test cross-section
    plot_cross_section(u_true, u_pred, axis='x', position=0.0, title="Test: ")
    
    # Test loss curves
    dummy_loss = [{'total_loss': 1.0 / (i+1), 'pde_loss': 0.5 / (i+1)} for i in range(100)]
    plot_loss_curves(dummy_loss, "Test Loss Evolution")
    
    print("Visualization tests completed!")