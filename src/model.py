import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Sequence, Dict, List
import numpy as np


class PINN(nn.Module):
    """Simple MLP-based Physics-Informed Neural Network for 2D problems.
    
    A standard feedforward neural network with customizable architecture
    for solving 2D partial differential equations using physics-informed training.
    """
    
    def __init__(self, features: Sequence[int], activation: str = "tanh", pos_enc: int = 0):
        super(PINN, self).__init__()
        self.features = features
        self.activation = activation
        self.pos_enc = pos_enc
        
        # Calculate input dimension after positional encoding
        if pos_enc > 0:
            # Each coordinate gets 2*pos_enc features (sin + cos for each frequency)
            # But the actual number of frequencies is different due to the range calculation
            num_freqs = len(range(int(-(pos_enc-1)/2), int((pos_enc+1)/2)))
            self.encoded_input_dim = 2 * 2 * num_freqs  # 2D input (x, y) with encoding
        else:
            self.encoded_input_dim = 2  # 2D input (x, y)
        
        # Build the network layers
        layers = []
        # First layer takes encoded input dimension
        layers.append(nn.Linear(self.encoded_input_dim, features[0]))
        layers.append(self._get_activation())
        
        # Hidden layers
        for i in range(len(features) - 1):
            layers.append(nn.Linear(features[i], features[i + 1]))
            if i < len(features) - 2:  # No activation on last layer
                layers.append(self._get_activation())
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _get_activation(self):
        """Get the activation function."""
        if self.activation == "tanh":
            return nn.Tanh()
        elif self.activation == "relu":
            return nn.ReLU()
        elif self.activation == "silu":
            return nn.SiLU()
        elif self.activation == "gelu":
            return nn.GELU()
        else:
            return nn.Tanh()  # Default fallback
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier normal initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def positional_encoding(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Apply positional encoding to 2D inputs (x, y)."""
        if self.pos_enc == 0:
            return torch.cat([x, y], dim=1)
        
        freq = torch.tensor([[2**k for k in range(int(-(self.pos_enc-1)/2), int((self.pos_enc+1)/2))]], 
                           dtype=x.dtype, device=x.device)
        
        # Apply encoding to both x and y
        x_enc = torch.cat([torch.sin(x @ freq), torch.cos(x @ freq)], dim=1)
        y_enc = torch.cat([torch.sin(y @ freq), torch.cos(y @ freq)], dim=1)
        
        return torch.cat([x_enc, y_enc], dim=1)
    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.
        
        Args:
            x: x-coordinates (N, 1)
            y: y-coordinates (N, 1)
            
        Returns:
            Function values (N, 1)
        """
        # Apply positional encoding if specified
        if self.pos_enc != 0:
            X = self.positional_encoding(x, y)
        else:
            X = torch.cat([x, y], dim=1)
        
        # Pass through the network
        return self.network(X)


def fn_evaluate(x: torch.Tensor, y: torch.Tensor, model: PINN) -> torch.Tensor:
    """Evaluate the PINN model at points (x, y).
    
    Args:
        x: x-coordinates (N, 1)
        y: y-coordinates (N, 1)
        model: PINN model instance
        
    Returns:
        Function values (N,)
    """
    return model(x, y).squeeze()


def fn_derivatives(x: torch.Tensor, y: torch.Tensor, model: PINN) -> Dict:
    """Compute derivatives of the PINN model at points (x, y) using PyTorch autograd.
    
    Args:
        x: x-coordinates (N, 1)
        y: y-coordinates (N, 1)
        model: PINN model instance
        
    Returns:
        Dictionary containing:
        - 'u': Function values
        - 'u_x': First derivative w.r.t. x
        - 'u_y': First derivative w.r.t. y
        - 'u_xx': Second derivative w.r.t. x
        - 'u_yy': Second derivative w.r.t. y
        - 'u_xy': Mixed derivative
    """
    # Ensure gradients are enabled
    x.requires_grad_(True)
    y.requires_grad_(True)
    
    # Function values
    u = fn_evaluate(x, y, model)
    
    # First derivatives
    u_x = torch.autograd.grad(
        u.sum(), x, create_graph=True, retain_graph=True
    )[0]
    
    u_y = torch.autograd.grad(
        u.sum(), y, create_graph=True, retain_graph=True
    )[0]
    
    # Second derivatives
    u_xx = torch.autograd.grad(
        u_x.sum(), x, create_graph=True, retain_graph=True
    )[0]
    
    u_yy = torch.autograd.grad(
        u_y.sum(), y, create_graph=True, retain_graph=True
    )[0]
    
    u_xy = torch.autograd.grad(
        u_x.sum(), y, create_graph=True, retain_graph=True
    )[0]
    
    return {
        'u': u,
        'u_x': u_x.squeeze(),
        'u_y': u_y.squeeze(),
        'u_xx': u_xx.squeeze(),
        'u_yy': u_yy.squeeze(),
        'u_xy': u_xy.squeeze()
    }




# ================= Example usage =================
if __name__ == "__main__":
    torch.manual_seed(0)
    
    print("=== PINN Model Examples (PyTorch - 2D Only) ===")
    
    # Create 2D grid
    print("\n1. 2D Example:")
    x = torch.rand(10, 1) * 2 - 1  # Random values in [-1, 1]
    y = torch.rand(10, 1) * 2 - 1
    
    model = PINN(features=[32, 32, 16, 1], activation="tanh", pos_enc=0)
    
    # Evaluate function
    u_vals = fn_evaluate(x, y, model)
    print(f"Function values shape: {u_vals.shape}")
    
    # Compute derivatives
    derivatives = fn_derivatives(x, y, model)
    print(f"Available derivatives: {list(derivatives.keys())}")
    print(f"u_x shape: {derivatives['u_x'].shape}")
    print(f"u_xx shape: {derivatives['u_xx'].shape}")
    
    # PDE residual example
    print("\n2. PDE Residual (Poisson equation):")
    residual = compute_pde_residual(x, y, model, "poisson")
    print(f"PDE residual shape: {residual.shape}")
    print(f"Mean residual: {torch.mean(torch.abs(residual)):.2e}")
    
    # Different activation functions
    print("\n3. Different Activations:")
    for activation in ["tanh", "relu", "silu", "gelu"]:
        model_act = PINN(features=[16, 16, 1], activation=activation, pos_enc=0)
        u_act = fn_evaluate(x, y, model_act)
        print(f"  {activation}: output range [{torch.min(u_act):.3f}, {torch.max(u_act):.3f}]")
    
    # Positional encoding example
    print("\n4. Positional Encoding:")
    model_pe = PINN(features=[32, 16, 1], activation="tanh", pos_enc=4)
    u_pe = fn_evaluate(x, y, model_pe)
    print(f"With PE: output range [{torch.min(u_pe):.3f}, {torch.max(u_pe):.3f}]")
    
    print("\n=== PINN: Simple 2D MLP with PyTorch autograd! ===")
