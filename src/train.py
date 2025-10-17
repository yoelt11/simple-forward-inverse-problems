import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from .dataloader import load_helmholtz_dataset, HelmholtzDataset
from .model import PINN, fn_derivatives


def create_collocation_points(n_points: int = 2500, domain: Tuple[float, float] = (-1, 1)) -> torch.Tensor:
    """Create random collocation points for PDE training."""
    x = torch.rand(n_points, 1) * (domain[1] - domain[0]) + domain[0]
    y = torch.rand(n_points, 1) * (domain[1] - domain[0]) + domain[0]
    return torch.cat([x, y], dim=1)


def compute_l2_error(pred: torch.Tensor, true: torch.Tensor) -> float:
    """Compute L2 relative error."""
    return torch.norm(pred - true) / torch.norm(true)


class ForwardTrainer:
    """Trainer for forward problems: learn u(x,y) given known parameters a1, a2, k"""
    
    def __init__(self, model: PINN, learning_rate: float = 1e-3):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.loss_history = []
        
    def train_step(self, X_collocation: torch.Tensor, sample_params: Dict,
                   bc_weight: float = 1.0, pde_weight: float = 1.0) -> Dict:
        """Single training step for forward problem."""
        self.optimizer.zero_grad()
        
        # Get derivatives
        derivs = fn_derivatives(X_collocation[:, 0:1], X_collocation[:, 1:2], self.model)
        u, u_xx, u_yy = derivs['u'], derivs['u_xx'], derivs['u_yy']
        x, y = X_collocation[:, 0], X_collocation[:, 1]
        
        # Extract parameters
        a1, a2, k = sample_params['a1'], sample_params['a2'], sample_params['k']
        
        # PDE loss: �u + k�u = q(x,y)
        q_coeff = -(a1 * np.pi)**2 - (a2 * np.pi)**2 + k**2
        q = q_coeff * torch.sin(a1 * np.pi * x) * torch.sin(a2 * np.pi * y)
        pde_residual = u_xx + u_yy + k**2 * u - q
        pde_loss = torch.mean(pde_residual**2)
        
        # Boundary conditions: u = sin(a1*�*x) * sin(a2*�*y) on boundaries
        # Sample boundary points
        n_bc = 100
        
        # Create boundary points
        boundary_points = []
        # Bottom/top boundaries
        x_boundary = torch.rand(n_bc//2, 1) * 2 - 1
        boundary_points.append(torch.cat([x_boundary, torch.full_like(x_boundary, -1)], dim=1))  # bottom
        boundary_points.append(torch.cat([x_boundary, torch.full_like(x_boundary, 1)], dim=1))   # top
        # Left/right boundaries  
        y_boundary = torch.rand(n_bc//2, 1) * 2 - 1
        boundary_points.append(torch.cat([torch.full_like(y_boundary, -1), y_boundary], dim=1))  # left
        boundary_points.append(torch.cat([torch.full_like(y_boundary, 1), y_boundary], dim=1))   # right
        
        bc_loss = 0.0
        for bc_points in boundary_points:
            bc_derivs = fn_derivatives(bc_points[:, 0:1], bc_points[:, 1:2], self.model)
            u_pred = bc_derivs['u']
            u_exact = torch.sin(a1 * np.pi * bc_points[:, 0]) * torch.sin(a2 * np.pi * bc_points[:, 1])
            bc_loss += torch.mean((u_pred - u_exact)**2)
        
        # Total loss
        total_loss = pde_weight * pde_loss + bc_weight * bc_loss
        total_loss.backward()
        self.optimizer.step()
        
        return {
            'total_loss': total_loss.item(),
            'pde_loss': pde_loss.item(),
            'bc_loss': bc_loss.item()
        }
    
    def train(self, dataset: HelmholtzDataset, sample_idx: int, n_epochs: int = 5000,
              n_collocation: int = 2500, bc_weight: float = 1.0, pde_weight: float = 1.0,
              print_every: int = 500) -> Dict:
        """Train on a single sample from the dataset."""
        
        sample = dataset.get_sample(sample_idx)
        sample_params = sample['params']
        
        print(f"Training forward problem for sample {sample_idx}")
        print(f"Parameters: a1={sample_params['a1']:.3f}, a2={sample_params['a2']:.3f}, k={sample_params['k']:.3f}")
        
        self.loss_history = []
        
        for epoch in range(n_epochs):
            # Create fresh collocation points each epoch
            X_collocation = create_collocation_points(n_collocation)
            
            losses = self.train_step(X_collocation, sample_params, bc_weight, pde_weight)
            self.loss_history.append(losses)
            
            if epoch % print_every == 0:
                print(f"Epoch {epoch}: Loss = {losses['total_loss']:.2e}, "
                      f"PDE = {losses['pde_loss']:.2e}, BC = {losses['bc_loss']:.2e}")
        
        # Compute final validation error
        # Evaluate on full grid (need gradients for fn_derivatives)
        X_grid = sample['X_coords'].clone()
        X_grid.requires_grad_(True)
        
        with torch.no_grad():
            self.model.eval()
            u_pred = self.model(X_grid[:, 0:1], X_grid[:, 1:2]).squeeze()
            u_pred = u_pred.reshape(sample['grid_shape'])
            u_true = sample['solution']
            l2_error = compute_l2_error(u_pred, u_true)
        
        self.model.train()
        
        print(f"Final L2 error: {l2_error:.2e}")
        
        return {
            'loss_history': self.loss_history,
            'final_l2_error': l2_error.item(),
            'u_pred': u_pred.detach(),
            'u_true': u_true
        }


class InverseTrainer:
    """Trainer for inverse problems: learn parameters a1, a2 given observation data"""
    
    def __init__(self, model: PINN, learning_rate: float = 1e-3, param_lr: float = 1e-2):
        self.model = model
        self.model_optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
        
        # Learnable parameters (initialized as random guesses)
        self.a1 = nn.Parameter(torch.tensor(2.0, requires_grad=True))
        self.a2 = nn.Parameter(torch.tensor(2.0, requires_grad=True)) 
        self.param_optimizer = optim.AdamW([self.a1, self.a2], lr=param_lr)
        
        self.loss_history = []
        self.param_history = []
        
    def train_step(self, X_collocation: torch.Tensor, X_obs: torch.Tensor, 
                   u_obs: torch.Tensor, k: float, data_weight: float = 10.0, 
                   pde_weight: float = 1.0) -> Dict:
        """Single training step for inverse problem."""
        self.model_optimizer.zero_grad()
        self.param_optimizer.zero_grad()
        
        # PDE loss on collocation points
        derivs = fn_derivatives(X_collocation[:, 0:1], X_collocation[:, 1:2], self.model)
        u, u_xx, u_yy = derivs['u'], derivs['u_xx'], derivs['u_yy']
        x, y = X_collocation[:, 0], X_collocation[:, 1]
        
        # Source term with learnable parameters
        q_coeff = -(self.a1 * np.pi)**2 - (self.a2 * np.pi)**2 + k**2
        q = q_coeff * torch.sin(self.a1 * np.pi * x) * torch.sin(self.a2 * np.pi * y)
        pde_residual = u_xx + u_yy + k**2 * u - q
        pde_loss = torch.mean(pde_residual**2)
        
        # Data fitting loss on observation points
        obs_derivs = fn_derivatives(X_obs[:, 0:1], X_obs[:, 1:2], self.model)
        u_pred = obs_derivs['u']
        data_loss = torch.mean((u_pred - u_obs)**2)
        
        # Total loss
        total_loss = pde_weight * pde_loss + data_weight * data_loss
        total_loss.backward()
        
        self.model_optimizer.step()
        self.param_optimizer.step()
        
        return {
            'total_loss': total_loss.item(),
            'pde_loss': pde_loss.item(),
            'data_loss': data_loss.item(),
            'a1_pred': self.a1.item(),
            'a2_pred': self.a2.item()
        }
    
    def train(self, dataset: HelmholtzDataset, sample_idx: int, n_epochs: int = 3000,
              n_collocation: int = 2000, n_obs: int = 200, data_weight: float = 10.0,
              pde_weight: float = 1.0, noise_level: float = 0.01, print_every: int = 300) -> Dict:
        """Train to discover parameters from observation data."""
        
        sample = dataset.get_sample(sample_idx)
        true_params = sample['params']
        k = true_params['k']  # Assume k is known
        
        # Get observation data with boundary focus
        X_obs, u_obs = dataset.get_observation_data(sample_idx, n_obs, noise_level, boundary_ratio=0.8)
        
        print(f"Training inverse problem for sample {sample_idx}")
        print(f"True parameters: a1={true_params['a1']:.3f}, a2={true_params['a2']:.3f}")
        print(f"Initial guesses: a1={self.a1.item():.3f}, a2={self.a2.item():.3f}")
        print(f"Using {n_obs} observations with {noise_level:.1%} noise")
        
        self.loss_history = []
        self.param_history = []
        
        for epoch in range(n_epochs):
            # Create fresh collocation points
            X_collocation = create_collocation_points(n_collocation)
            
            losses = self.train_step(X_collocation, X_obs, u_obs, k, data_weight, pde_weight)
            self.loss_history.append(losses)
            self.param_history.append((losses['a1_pred'], losses['a2_pred']))
            
            if epoch % print_every == 0:
                print(f"Epoch {epoch}: Loss = {losses['total_loss']:.2e}, "
                      f"a1 = {losses['a1_pred']:.3f} (true: {true_params['a1']:.3f}), "
                      f"a2 = {losses['a2_pred']:.3f} (true: {true_params['a2']:.3f})")
        
        # Final parameter errors
        a1_error = abs(self.a1.item() - true_params['a1']) / true_params['a1']
        a2_error = abs(self.a2.item() - true_params['a2']) / true_params['a2']
        
        print(f"Final parameter errors: a1 = {a1_error:.1%}, a2 = {a2_error:.1%}")
        
        return {
            'loss_history': self.loss_history,
            'param_history': self.param_history,
            'final_params': {'a1': self.a1.item(), 'a2': self.a2.item()},
            'true_params': true_params,
            'param_errors': {'a1': a1_error, 'a2': a2_error}
        }


# Example usage
if __name__ == "__main__":
    # Load dataset
    dataset = load_helmholtz_dataset()
    
    print("=== Forward Problem Example ===")
    # Forward problem: learn solution given known parameters
    model_fwd = PINN([50, 50, 50, 1], activation="tanh", pos_enc=0)
    trainer_fwd = ForwardTrainer(model_fwd, learning_rate=1e-3)
    
    results_fwd = trainer_fwd.train(dataset, sample_idx=0, n_epochs=2000, print_every=400)
    
    print(f"\n=== Inverse Problem Example ===")
    # Inverse problem: discover parameters from data
    model_inv = PINN([50, 50, 50, 1], activation="tanh", pos_enc=0)  
    trainer_inv = InverseTrainer(model_inv, learning_rate=1e-3, param_lr=1e-2)
    
    results_inv = trainer_inv.train(dataset, sample_idx=0, n_epochs=1500, print_every=300)