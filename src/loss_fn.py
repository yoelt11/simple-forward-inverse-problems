import torch
from typing import Any, Optional, Tuple, Dict
from .model import fn_derivatives


def analytical_solution(x: torch.Tensor, y: torch.Tensor, a1: float, a2: float) -> torch.Tensor:
    """Analytical solution for the Helmholtz equation."""
    return torch.sin(a1 * torch.pi * x) * torch.sin(a2 * torch.pi * y)


def compute_source_term(x: torch.Tensor, y: torch.Tensor, k: float, a1: float, a2: float) -> torch.Tensor:
    """Manufactured source term for the Helmholtz equation."""
    q_coeff = -(a1 * torch.pi)**2 - (a2 * torch.pi)**2 + k**2
    return q_coeff * torch.sin(a1 * torch.pi * x) * torch.sin(a2 * torch.pi * y)


def compute_boundary_loss(u_reshaped: torch.Tensor, x_coords: torch.Tensor, y_coords: torch.Tensor, 
                         a1: float, a2: float) -> torch.Tensor:
    """Compute Dirichlet boundary condition loss for all four boundaries."""
    boundaries = [
        (u_reshaped[0, :], x_coords[0, :], y_coords[0, :]),    # bottom
        (u_reshaped[-1, :], x_coords[-1, :], y_coords[-1, :]), # top
        (u_reshaped[:, 0], x_coords[:, 0], y_coords[:, 0]),    # left
        (u_reshaped[:, -1], x_coords[:, -1], y_coords[:, -1])  # right
    ]
    
    total_bc_loss = 0.0
    for u_pred, x_bc, y_bc in boundaries:
        u_exact = analytical_solution(x_bc, y_bc, a1, a2)
        total_bc_loss += torch.mean((u_pred - u_exact)**2)
    
    return total_bc_loss


def helmholtz_2d_loss_fwd(X: torch.Tensor, model: Any,
                      k: float = 1.0, a1: float = 1.0, a2: float = 1.0,
                      bc_weight: float = 1.0, pde_weight: float = 1.0,
                      grid_shape: Optional[Tuple[int, int]] = None,
                      return_components: bool = False) -> torch.Tensor:
    """Compute loss for 2D Helmholtz equation: Δu + k²u = q(x,y)
    
    Exact solution: u(x,y) = sin(a₁πx) sin(a₂πy) on domain [-1,1] × [-1,1]
    """
    # Get derivatives and coordinates  
    derivs = fn_derivatives(X[:, 0:1], X[:, 1:2], model)
    u, u_xx, u_yy = derivs["u"], derivs["u_xx"], derivs["u_yy"]
    x, y = X[:, 0], X[:, 1]
    
    # PDE loss: Δu + k²u - q = 0
    q = compute_source_term(x, y, k, a1, a2)
    pde_residual = u_xx + u_yy + k**2 * u - q
    pde_loss = torch.mean(pde_residual**2)
    
    # Boundary condition loss
    if grid_shape is None:
        n_sqrt = int(torch.sqrt(torch.tensor(X.shape[0], dtype=torch.float32)))
        grid_shape = (n_sqrt, n_sqrt)
    
    n_x, n_y = grid_shape
    u_reshaped = u.reshape(n_y, n_x)
    x_coords = X[:, 0].reshape(n_y, n_x)
    y_coords = X[:, 1].reshape(n_y, n_x)
    
    bc_loss = compute_boundary_loss(u_reshaped, x_coords, y_coords, a1, a2)
    
    # Combine losses
    total_loss = pde_weight * pde_loss + bc_weight * bc_loss
    
    if return_components:
        return {
            'total_loss': total_loss,
            'pde_loss': pde_weight * pde_loss,
            'bc_loss': bc_weight * bc_loss,
            'unweighted_pde_loss': pde_loss,
            'unweighted_bc_loss': bc_loss
        }
    
    return total_loss


def helmholtz_2d_loss_inv(X: torch.Tensor, model: Any,
                         a1: torch.Tensor, a2: torch.Tensor, 
                         k: float = 1.0,
                         data_obs: torch.Tensor = None,
                         X_obs: torch.Tensor = None,
                         data_weight: float = 1.0, 
                         pde_weight: float = 1.0,
                         return_components: bool = False) -> torch.Tensor:
    """Inverse problem loss: learn parameters a1, a2 from observed data.
    
    Args:
        X: Collocation points for PDE (N, 2)
        a1, a2: Learnable frequency parameters (scalars as torch tensors)
        data_obs: Observed solution data (M,)
        X_obs: Observation points (M, 2) 
        data_weight: Weight for data fitting loss
        pde_weight: Weight for PDE physics loss
    """
    # PDE loss on collocation points
    derivs = fn_derivatives(X[:, 0:1], X[:, 1:2], model)
    u, u_xx, u_yy = derivs["u"], derivs["u_xx"], derivs["u_yy"]
    x, y = X[:, 0], X[:, 1]
    
    q = compute_source_term(x, y, k, a1, a2)
    pde_residual = u_xx + u_yy + k**2 * u - q
    pde_loss = torch.mean(pde_residual**2)
    
    # Data fitting loss on observation points
    data_loss = 0.0
    if data_obs is not None and X_obs is not None:
        derivs_obs = fn_derivatives(X_obs[:, 0:1], X_obs[:, 1:2], model)
        u_pred = derivs_obs["u"]
        data_loss = torch.mean((u_pred - data_obs)**2)
    
    total_loss = pde_weight * pde_loss + data_weight * data_loss
    
    if return_components:
        return {
            'total_loss': total_loss,
            'pde_loss': pde_weight * pde_loss,
            'data_loss': data_weight * data_loss,
            'unweighted_pde_loss': pde_loss,
            'unweighted_data_loss': data_loss
        }
    
    return total_loss