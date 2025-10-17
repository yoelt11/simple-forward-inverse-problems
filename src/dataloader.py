import pickle
import torch
from typing import Tuple, Dict
import os


class HelmholtzDataset:
    """Simple dataloader for Helmholtz 2D dataset.
    
    For forward problems: use ground truth parameters to validate predictions
    For inverse problems: use solution data to learn unknown parameters
    """
    
    def __init__(self, dataset_path: str):
        """Load the dataset from pickle file."""
        with open(dataset_path, 'rb') as f:
            self.data = pickle.load(f)
        
        # Extract key components
        self.solutions = torch.tensor(self.data['solutions'], dtype=torch.float32)  # (200, 64, 64)
        self.params = self.data['pde_params']
        self.a1 = torch.tensor(self.params['a1'], dtype=torch.float32)  # (200,)
        self.a2 = torch.tensor(self.params['a2'], dtype=torch.float32)  # (200,)
        self.k = torch.tensor(self.params['k'], dtype=torch.float32)    # (200,)
        
        # Coordinate grids
        self.X_grid = torch.tensor(self.data['X_grid'], dtype=torch.float32)  # (64, 64)
        self.Y_grid = torch.tensor(self.data['Y_grid'], dtype=torch.float32)  # (64, 64)
        
        # Flatten coordinates for loss function compatibility
        self.X_flat = torch.stack([self.X_grid.flatten(), self.Y_grid.flatten()], dim=1)  # (4096, 2)
        
        self.n_samples = self.solutions.shape[0]
        self.grid_shape = self.solutions.shape[1:]  # (64, 64)
        
        print(f"Loaded dataset: {self.n_samples} samples, grid {self.grid_shape}")
        print(f"Parameter ranges:")
        print(f"  a1: [{self.a1.min():.3f}, {self.a1.max():.3f}]")
        print(f"  a2: [{self.a2.min():.3f}, {self.a2.max():.3f}]")
        print(f"  k:  [{self.k.min():.3f}, {self.k.max():.3f}]")
    
    def get_sample(self, idx: int) -> Dict:
        """Get a single sample by index.
        
        Returns:
            Dictionary with:
            - 'solution': Ground truth solution (64, 64)
            - 'solution_flat': Flattened solution (4096,)
            - 'params': {'a1': float, 'a2': float, 'k': float}
            - 'X_coords': Coordinate points (4096, 2)
            - 'grid_shape': (64, 64)
        """
        return {
            'solution': self.solutions[idx],
            'solution_flat': self.solutions[idx].flatten(),
            'params': {
                'a1': float(self.a1[idx]),
                'a2': float(self.a2[idx]), 
                'k': float(self.k[idx])
            },
            'X_coords': self.X_flat,
            'grid_shape': self.grid_shape
        }
    
    def get_observation_data(self, idx: int, n_obs: int = 100, 
                           noise_level: float = 0.0, boundary_ratio: float = 0.7) -> Tuple:
        """Get sparse observation data for inverse problems with boundary focus.
        
        Args:
            idx: Sample index
            n_obs: Number of observation points
            noise_level: Add Gaussian noise (std) to observations
            boundary_ratio: Fraction of observations near boundaries (0.0-1.0)
            
        Returns:
            (X_obs, u_obs): Observation points and corresponding solution values
        """
        ny, nx = self.grid_shape
        n_boundary = int(n_obs * boundary_ratio)
        n_interior = n_obs - n_boundary
        
        # Get boundary indices (edges of the grid)
        boundary_indices = []
        
        # Top and bottom boundaries
        for j in range(nx):
            boundary_indices.append(0 * nx + j)      # bottom
            boundary_indices.append((ny-1) * nx + j) # top
        
        # Left and right boundaries (excluding corners already added)
        for i in range(1, ny-1):
            boundary_indices.append(i * nx + 0)      # left
            boundary_indices.append(i * nx + (nx-1)) # right
        
        boundary_indices = torch.tensor(boundary_indices)
        
        # Sample boundary observations
        if n_boundary > 0 and len(boundary_indices) > 0:
            if n_boundary >= len(boundary_indices):
                # Use all boundary points
                boundary_obs_indices = boundary_indices
            else:
                # Sample from boundary points
                perm = torch.randperm(len(boundary_indices))[:n_boundary]
                boundary_obs_indices = boundary_indices[perm]
        else:
            boundary_obs_indices = torch.tensor([], dtype=torch.long)
        
        # Sample interior observations
        if n_interior > 0:
            # Get all interior indices (not on boundary)
            all_indices = torch.arange(self.X_flat.shape[0])
            interior_mask = torch.ones(len(all_indices), dtype=torch.bool)
            interior_mask[boundary_indices] = False
            interior_indices = all_indices[interior_mask]
            
            if n_interior >= len(interior_indices):
                interior_obs_indices = interior_indices
            else:
                perm = torch.randperm(len(interior_indices))[:n_interior]
                interior_obs_indices = interior_indices[perm]
        else:
            interior_obs_indices = torch.tensor([], dtype=torch.long)
        
        # Combine boundary and interior observations
        obs_indices = torch.cat([boundary_obs_indices, interior_obs_indices])
        
        X_obs = self.X_flat[obs_indices]
        u_obs = self.solutions[idx].flatten()[obs_indices]
        
        # Add noise if specified
        if noise_level > 0.0:
            noise = noise_level * torch.randn_like(u_obs)
            u_obs = u_obs + noise
        
        return X_obs, u_obs


def load_helmholtz_dataset(dataset_path: str = None) -> HelmholtzDataset:
    """Convenience function to load the dataset."""
    if dataset_path is None:
        # Get the directory where this script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Go up one level to project root, then to dataset
        project_root = os.path.dirname(script_dir)
        dataset_path = os.path.join(project_root, "dataset", "helmholtz2D", "ground_truth", "helmholtz2d_dataset.pkl")
    
    return HelmholtzDataset(dataset_path)


# Example usage
if __name__ == "__main__":
    # Load dataset
    dataset = load_helmholtz_dataset()
    
    # Forward problem example: get full solution for validation
    sample_0 = dataset.get_sample(0)
    print(f"\nSample 0 parameters: {sample_0['params']}")
    print(f"Solution shape: {sample_0['solution'].shape}")
    
    # Inverse problem example: get sparse observations  
    X_obs, u_obs = dataset.get_observation_data(0, n_obs=50, noise_level=0.01)
    print(f"\nObservation data shape: {X_obs.shape}, {u_obs.shape}")
    print(f"Observation points range: x=[{X_obs[:, 0].min():.2f}, {X_obs[:, 0].max():.2f}]")
    print(f"Observation values range: u=[{u_obs.min():.3f}, {u_obs.max():.3f}]")