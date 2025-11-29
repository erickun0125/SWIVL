"""
High-Level Policy Training Script

Trains high-level policies (Flow Matching, Diffusion, ACT) for bimanual manipulation
using vision-based imitation learning from demonstrations.

Usage:
    python scripts/training/train_hl_policy.py --policy flow_matching --config configs/rl_config.yaml
    python scripts/training/train_hl_policy.py --policy diffusion --epochs 200
    python scripts/training/train_hl_policy.py --policy act --batch_size 128

Example:
    # Train Flow Matching policy
    python scripts/training/train_hl_policy.py --policy flow_matching

    # Train Diffusion Policy with custom dataset
    python scripts/training/train_hl_policy.py --policy diffusion --dataset ./data/custom

    # Train ACT with custom config
    python scripts/training/train_hl_policy.py --policy act --config my_config.yaml
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import yaml

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.hl_planners.flow_matching import FlowMatchingPolicy
from src.hl_planners.diffusion_policy import DiffusionPolicy
from src.hl_planners.act import ACTPolicy


class LinearNormalizer:
    """
    Min-Max Normalizer for input and output data.
    Scales data to [0, 1] or [-1, 1].
    """
    def __init__(self):
        self.stats = {}

    def fit(self, data: np.ndarray, key: str):
        """Compute min/max stats."""
        self.stats[key] = {
            'min': np.min(data, axis=0),
            'max': np.max(data, axis=0)
        }

    def normalize(self, data: np.ndarray, key: str) -> np.ndarray:
        """Normalize data using stored stats."""
        if key not in self.stats:
            return data
        stats = self.stats[key]
        # Scale to [-1, 1] which is better for neural nets than [0, 1]
        # x_norm = 2 * (x - x_min) / (x_max - x_min) - 1
        denominator = stats['max'] - stats['min']
        denominator[denominator == 0] = 1.0
        return 2 * (data - stats['min']) / denominator - 1

    def denormalize(self, data: np.ndarray, key: str) -> np.ndarray:
        """Denormalize data using stored stats."""
        if key not in self.stats:
            return data
        stats = self.stats[key]
        # x = (x_norm + 1) / 2 * (x_max - x_min) + x_min
        denominator = stats['max'] - stats['min']
        return (data + 1) / 2 * denominator + stats['min']
    
    def state_dict(self):
        return self.stats
    
    def load_state_dict(self, state_dict):
        self.stats = state_dict


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_policy(policy_type: str, config: Dict[str, Any], device: torch.device):
    """
    Create high-level policy based on type.

    Args:
        policy_type: Type of policy (flow_matching, diffusion, act)
        config: Configuration dictionary
        device: Device for computation

    Returns:
        Policy instance
    """
    print(f"Creating {policy_type} policy...")
    
    # Extract horizons from config or use defaults
    # These must match the dataset generation parameters
    obs_horizon = config.get('model', {}).get('obs_horizon', 1)
    pred_horizon = config.get('model', {}).get('pred_horizon', 10) # Default 10
    use_external_wrench = config.get('model', {}).get('use_external_wrench', True)

    if policy_type == 'flow_matching':
        from src.hl_planners.flow_matching import FlowMatchingConfig
        policy_config = FlowMatchingConfig(
            context_length=obs_horizon,
            action_dim=6, 
            pred_horizon=pred_horizon,
            use_external_wrench=use_external_wrench
        )
        policy = FlowMatchingPolicy(config=policy_config, device=device)
        
    elif policy_type == 'diffusion':
        from src.hl_planners.diffusion_policy import DiffusionPolicyConfig
        policy_config = DiffusionPolicyConfig(
            obs_horizon=obs_horizon,
            pred_horizon=pred_horizon,
            action_dim=6,
            use_external_wrench=use_external_wrench
        )
        policy = DiffusionPolicy(config=policy_config, device=device)
        
    elif policy_type == 'act':
        from src.hl_planners.act import ACTConfig
        policy_config = ACTConfig(
            obs_horizon=obs_horizon,
            pred_horizon=pred_horizon,
            action_dim=6,
            use_external_wrench=use_external_wrench
        )
        policy = ACTPolicy(config=policy_config, device=device)
        
    else:
        raise ValueError(f"Unknown policy type: {policy_type}")

    print(f"✓ Policy created with {sum(p.numel() for p in policy.parameters())} parameters")
    return policy


def create_optimizer(policy, config: Dict[str, Any]):
    """Create optimizer for training."""
    optimizer_type = config.get('training', {}).get('optimizer', 'adam').lower()
    lr = config.get('training', {}).get('learning_rate', 1e-4)
    weight_decay = config.get('training', {}).get('weight_decay', 1e-5)

    if optimizer_type == 'adam':
        optimizer = optim.Adam(policy.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type == 'adamw':
        optimizer = optim.AdamW(policy.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type == 'sgd':
        optimizer = optim.SGD(policy.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")

    return optimizer


def create_scheduler(optimizer, config: Dict[str, Any], num_epochs: int):
    """Create learning rate scheduler."""
    scheduler_config = config.get('training', {}).get('lr_scheduler', {})
    scheduler_type = scheduler_config.get('type', 'none').lower()

    if scheduler_type == 'cosine':
        warmup_epochs = scheduler_config.get('warmup_epochs', 10)
        min_lr = scheduler_config.get('min_lr', 1e-6)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=num_epochs - warmup_epochs,
            eta_min=min_lr
        )
    elif scheduler_type == 'step':
        step_size = scheduler_config.get('step_size', 30)
        gamma = scheduler_config.get('gamma', 0.1)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    else:
        scheduler = None

    return scheduler


def load_dataset(dataset_path: str, config: Dict[str, Any]):
    """
    Load demonstration dataset.
    
    Assumes dataset is stored in HDF5 format with structure:
    demo_0/
        obs/
            ee_poses: (T, 2, 3)
            link_poses: (T, 2, 3)
            external_wrenches: (T, 2, 3)
        action: (T, 2, 3)  <-- Desired poses
    """
    print(f"Loading dataset from {dataset_path}...")

    if dataset_path is None or not os.path.exists(dataset_path):
        print(f"Dataset path {dataset_path} does not exist. Creating dummy dataset.")
        # ... dummy dataset logic ... (kept as fallback)
        N = 100
        # Dummy images: (N, T, 3, 96, 96)
        pred_horizon = config.get('model', {}).get('pred_horizon', 10)
        T = pred_horizon
        action_dim = 6
        proprio_dim = 18 if config.get('model', {}).get('use_external_wrench', True) else 12
        
        images = np.random.randint(0, 255, (N, T, 3, 96, 96)).astype(np.uint8)
        proprio = np.random.randn(N, T, proprio_dim).astype(np.float32)
        action = np.random.randn(N, T, action_dim).astype(np.float32)
        
        class DummyDataset(torch.utils.data.Dataset):
            def __init__(self, images, proprio, action):
                self.images = torch.from_numpy(images)
                self.proprio = torch.from_numpy(proprio)
                self.action = torch.from_numpy(action)
            def __len__(self): return len(self.proprio)
            def __getitem__(self, idx): 
                return (self.images[idx].to(torch.float32) / 255.0, self.proprio[idx]), self.action[idx]
            
        dataset = DummyDataset(images, proprio, action)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        return random_split(dataset, [train_size, val_size]), LinearNormalizer()

    import h5py
    
    class BiArtHDF5Dataset(torch.utils.data.Dataset):
        def __init__(self, hdf5_path, obs_horizon=1, pred_horizon=10, use_external_wrench=True):
            """
            Args:
                hdf5_path: Path to .h5 file OR directory containing .h5 files
                obs_horizon: Number of past steps to stack
                pred_horizon: Number of future steps to predict
                use_external_wrench: Whether to include external wrenches in observation
            """
            super().__init__()
            self.obs_horizon = obs_horizon
            self.pred_horizon = pred_horizon
            self.use_external_wrench = use_external_wrench
            self.normalizer = LinearNormalizer()
            
            # Load all data into memory (for simplicity)
            self.episodes = []
            
            # Temporary buffers for computing stats
            all_proprio = []
            all_actions = []
            
            # Check if path is directory or file
            if os.path.isdir(hdf5_path):
                # Recursively find all .h5 files in directory and subdirectories
                import glob
                h5_pattern = os.path.join(hdf5_path, '**', '*.h5')
                file_paths = sorted(glob.glob(h5_pattern, recursive=True))
                print(f"Found {len(file_paths)} HDF5 files in {hdf5_path} (including subdirectories)")
                
                # Also check direct children (non-recursive) if glob found nothing
                if len(file_paths) == 0:
                    h5_files = sorted([f for f in os.listdir(hdf5_path) if f.endswith('.h5')])
                    file_paths = [os.path.join(hdf5_path, f) for f in h5_files]
                    print(f"Fallback: Found {len(file_paths)} HDF5 files directly in {hdf5_path}")
            else:
                # Single file
                file_paths = [hdf5_path]
                print(f"Loading single file: {hdf5_path}")
            
            # Load each file
            for file_path in file_paths:
                with h5py.File(file_path, 'r') as f:
                    num_demos = f.attrs.get('num_demos', 0)
                    print(f"  Loading {file_path}: {num_demos} demos")
                    
                    # Handle single episode file (direct data)
                    if num_demos == 0 and 'obs' in f and 'action' in f:
                        print("  Detected single episode file format.")
                        num_demos = 1
                        is_single_episode = True
                    else:
                        is_single_episode = False
                    
                    for i in range(num_demos):
                        if is_single_episode:
                            g = f
                        else:
                            demo_key = f'demo_{i}'
                            if demo_key not in f: continue
                            g = f[demo_key]
                        # Load obs
                        obs_group = g['obs']
                        ee_poses = obs_group['ee_poses'][:]  # (T, 2, 3)
                        link_poses = obs_group['link_poses'][:] # (T, 2, 3)
                        wrenches = obs_group['external_wrenches'][:] # (T, 2, 3)
                        
                        # Get sequence length from loaded data
                        T = ee_poses.shape[0]
                        
                        body_twists = obs_group.get('ee_body_twists')
                        if body_twists is not None:
                            body_twists = body_twists[:]
                        else:
                            body_twists = np.zeros_like(ee_poses)
                        
                        # Load images: (T, H, W, 3) -> (T, 3, H, W)
                        images = obs_group['images'][:] 
                        images = np.transpose(images, (0, 3, 1, 2))
                        
                        # Construct proprioception
                        # EE Poses (6) + EE Velocities (6) + [Optional] Wrenches (6)
                        # NOTE: ee_velocities are point velocities [vx, vy, omega], NOT twists
                        proprio_list = [
                            ee_poses.reshape(T, -1),
                            np.zeros((T, 6))  # Default fallback
                        ]
                        
                        # Load velocities from file (support both old and new naming)
                        if 'ee_velocities' in obs_group:
                            ee_velocities = obs_group['ee_velocities'][:]
                            proprio_list[1] = ee_velocities.reshape(T, -1)
                        elif 'ee_twists' in obs_group:
                            # Legacy support for old data files
                            ee_velocities = obs_group['ee_twists'][:]
                            proprio_list[1] = ee_velocities.reshape(T, -1)
                            
                        if self.use_external_wrench:
                            proprio_list.append(wrenches.reshape(T, -1))
                            
                        proprio = np.concatenate(proprio_list, axis=1)
                        
                        # Load action: (T, 2, 3) -> (T, 6)
                        action_group = g['action']
                        desired_poses = action_group['desired_poses'][:]  # (T, 2, 3)
                        action = desired_poses.reshape(T, -1)
                        
                        self.episodes.append({
                            'image': images, # Keep as numpy (uint8)
                            'proprio': proprio, # Keep as numpy
                            'action': action
                        })
                        all_proprio.append(proprio)
                        all_actions.append(action)
            
            # Compute stats
            print(f"\nTotal episodes loaded: {len(self.episodes)}")
            all_proprio_concat = np.concatenate(all_proprio, axis=0)
            all_actions_concat = np.concatenate(all_actions, axis=0)
            self.normalizer.fit(all_proprio_concat, 'proprio')
            self.normalizer.fit(all_actions_concat, 'action')
            print("Normalization stats computed.")
            
            # Normalize data in memory
            for ep in self.episodes:
                # Images: [0, 255] -> [0, 1]
                ep['image'] = torch.from_numpy(ep['image'].astype(np.float32) / 255.0)
                
                ep['proprio'] = torch.from_numpy(
                    self.normalizer.normalize(ep['proprio'], 'proprio').astype(np.float32)
                )
                ep['action'] = torch.from_numpy(
                    self.normalizer.normalize(ep['action'], 'action').astype(np.float32)
                )
            
            # Create indices
            self.indices = []
            for ep_idx, episode in enumerate(self.episodes):
                seq_len = len(episode['proprio'])
                # Valid start indices
                # Need enough history: i >= obs_horizon - 1
                # Need enough future: i + pred_horizon <= seq_len
                # But for simple 1->1 mapping policies:
                for i in range(obs_horizon - 1, seq_len - pred_horizon + 1):
                    self.indices.append((ep_idx, i))
                    
        def __len__(self):
            return len(self.indices)
            
        def __getitem__(self, idx):
            ep_idx, t = self.indices[idx]
            episode = self.episodes[ep_idx]
            
            # Get observation window
            # Assuming policy expects (obs_horizon, state_dim) or flattened
            start = t - self.obs_horizon + 1
            end = t + 1
            
            image_seq = episode['image'][start:end]  # (obs_horizon, 3, H, W)
            # NOTE: Images are already normalized to [0, 1] in __init__, no need to divide again!
            proprio_seq = episode['proprio'][start:end]  # (obs_horizon, proprio_dim)
            
            # Get action window
            # Assuming policy expects (pred_horizon, action_dim)
            # For simple diffusion/ACT, usually predict future chunk
            act_start = t
            act_end = t + self.pred_horizon
            action_seq = episode['action'][act_start:act_end] # (pred_horizon, action_dim)
            
            return (image_seq, proprio_seq), action_seq

    # Determine horizons based on config/policy type
    # For generic compatibility, default to horizons that match the policies
    
    # Safe defaults
    obs_h = config.get('model', {}).get('obs_horizon', 1)
    pred_h = config.get('model', {}).get('pred_horizon', 10)
    use_wrench = config.get('model', {}).get('use_external_wrench', True)
    
    print(f"Dataset Configuration:")
    print(f"  Observation Horizon: {obs_h}")
    print(f"  Prediction Horizon: {pred_h}")
    print(f"  External Wrench: {use_wrench}")
    
    # Create dataset
    full_dataset = BiArtHDF5Dataset(
        dataset_path, 
        obs_horizon=obs_h, 
        pred_horizon=pred_h,
        use_external_wrench=use_wrench
    )
    
    # Split
    val_ratio = 1.0 - config.get('dataset', {}).get('train_split', 0.8)
    val_size = int(val_ratio * len(full_dataset))
    train_size = len(full_dataset) - val_size
    
    return random_split(full_dataset, [train_size, val_size]), full_dataset.normalizer


def train_epoch(
    policy,
    train_loader,
    optimizer,
    device: torch.device,
    epoch: int,
    writer: Optional[SummaryWriter] = None,
    log_interval: int = 20
):
    """
    Train for one epoch.
    
    Args:
        policy: Policy model to train
        train_loader: Training data loader
        optimizer: Optimizer
        device: Device for computation
        epoch: Current epoch number
        writer: TensorBoard writer (optional)
        log_interval: How often to log batch loss (default: 20)
    
    Returns:
        Average training loss for the epoch
    """
    policy.train()
    total_loss = 0.0
    num_batches = 0

    for batch_idx, ((images, proprio), actions) in enumerate(train_loader):
        images = images.to(device)
        proprio = proprio.to(device)
        actions = actions.to(device)

        # Forward pass
        optimizer.zero_grad()
        loss = policy.compute_loss(images, proprio, actions)

        # Backward pass
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
        
        optimizer.step()

        # Logging
        total_loss += loss.item()
        num_batches += 1

        if batch_idx % log_interval == 0:
            print(f"  Batch [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.6f}")
            if writer is not None:
                global_step = epoch * len(train_loader) + batch_idx
                writer.add_scalar('train/batch_loss', loss.item(), global_step)

    avg_loss = total_loss / num_batches
    return avg_loss


def validate(policy, val_loader, device: torch.device):
    """Validate policy."""
    policy.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for (images, proprio), actions in val_loader:
            images = images.to(device)
            proprio = proprio.to(device)
            actions = actions.to(device)

            loss = policy.compute_loss(images, proprio, actions)
            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches
    return avg_loss


def train(
    policy_type: str,
    config_path: str = 'scripts/configs/rl_config.yaml',
    dataset_path: Optional[str] = None,
    num_epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    device: str = 'auto',
    checkpoint_dir: Optional[str] = None,
    resume_from: Optional[str] = None,
    obs_horizon: Optional[int] = None,
    pred_horizon: Optional[int] = None,
    use_external_wrench: Optional[bool] = None
):
    """
    Train high-level policy.

    Args:
        policy_type: Type of policy to train
        config_path: Path to configuration file
        dataset_path: Path to demonstration dataset
        num_epochs: Number of training epochs (overrides config)
        batch_size: Batch size (overrides config)
        device: Device for computation
        checkpoint_dir: Directory to save checkpoints
        resume_from: Path to checkpoint to resume from
    """
    print("=" * 80)
    print(f"Training {policy_type.upper()} Policy")
    print("=" * 80)

    # Load configuration
    config = load_config(config_path)

    # Determine device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch_device = torch.device(device)
    print(f"Using device: {torch_device}")

    # Override config with command-line arguments
    if dataset_path is None:
        dataset_path = config.get('dataset', {}).get('path', './data/demonstrations')
    if num_epochs is None:
        num_epochs = config.get('training', {}).get('num_epochs', 10000)
    if batch_size is None:
        batch_size = config.get('training', {}).get('batch_size', 64)
    if checkpoint_dir is None:
        checkpoint_dir = config.get('output', {}).get('checkpoint_dir', './checkpoints')
        
    # Apply overrides
    if obs_horizon is not None:
        config.setdefault('model', {})['obs_horizon'] = obs_horizon
    if pred_horizon is not None:
        config.setdefault('model', {})['pred_horizon'] = pred_horizon
    if use_external_wrench is not None:
        config.setdefault('model', {})['use_external_wrench'] = use_external_wrench

    # Create policy
    policy = create_policy(policy_type, config, torch_device)

    # Load dataset first to compute normalization stats
    (train_dataset, val_dataset), normalizer = load_dataset(dataset_path, config)

    if train_dataset is None:
        # ... error handling ...
        return

    # Set normalizer to policy if applicable
    if hasattr(policy, 'set_normalizer'):
        policy.set_normalizer(normalizer)

    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_loss = float('inf')
    if resume_from is not None and os.path.exists(resume_from):
        print(f"Resuming from checkpoint: {resume_from}")
        checkpoint = torch.load(resume_from, map_location=torch_device)
        policy.load_state_dict(checkpoint['model_state_dict'])
        
        # Load normalizer if available
        if 'normalizer' in checkpoint and hasattr(policy, 'set_normalizer'):
             normalizer.load_state_dict(checkpoint['normalizer'])
             policy.set_normalizer(normalizer)

        start_epoch = checkpoint.get('epoch', 0) + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        print(f"Resumed from epoch {start_epoch}")

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Create optimizer and scheduler
    optimizer = create_optimizer(policy, config)
    scheduler = create_scheduler(optimizer, config, num_epochs)

    # Create tensorboard writer
    log_dir = config.get('logging', {}).get('tensorboard_log', './logs/hl_policy/')
    writer = SummaryWriter(log_dir=os.path.join(log_dir, policy_type))

    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Training loop
    print("\n" + "=" * 80)
    print(f"Starting training for {num_epochs} epochs...")
    print("=" * 80)

    for epoch in range(start_epoch, num_epochs):
        print(f"\nEpoch [{epoch+1}/{num_epochs}]")

        # Train
        train_loss = train_epoch(
            policy, train_loader, optimizer, torch_device, epoch, writer,
            log_interval=config.get('logging', {}).get('log_interval', 20)
        )
        print(f"  Train Loss: {train_loss:.6f}")

        # Validate
        val_loss = validate(policy, val_loader, torch_device)
        print(f"  Val Loss: {val_loss:.6f}")

        # Log to tensorboard
        writer.add_scalar('train/epoch_loss', train_loss, epoch)
        writer.add_scalar('val/epoch_loss', val_loss, epoch)
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

        # Update learning rate
        if scheduler is not None:
            scheduler.step()

        # Save checkpoint
        save_freq = config.get('logging', {}).get('save_freq', 1000)
        if (epoch + 1) % save_freq == 0:
            checkpoint_path = os.path.join(
                checkpoint_dir,
                f"{policy_type}_epoch_{epoch+1}.pth"
            )
            save_dict = {
                'epoch': epoch,
                'model_state_dict': policy.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'best_val_loss': best_val_loss
            }
            if hasattr(policy, 'normalizer'):
                save_dict['normalizer'] = policy.normalizer.state_dict()
                
            torch.save(save_dict, checkpoint_path)
            print(f"  Saved checkpoint: {checkpoint_path}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(checkpoint_dir, f"{policy_type}_best.pth")
            save_dict = {
                'epoch': epoch,
                'model_state_dict': policy.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'best_val_loss': best_val_loss
            }
            if hasattr(policy, 'normalizer'):
                save_dict['normalizer'] = policy.normalizer.state_dict()

            torch.save(save_dict, best_model_path)
            print(f"  ✓ New best model saved (val_loss: {val_loss:.6f})")

    print("\n" + "=" * 80)
    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print("=" * 80)

    writer.close()


def main():
    parser = argparse.ArgumentParser(
        description='Train high-level policy for bimanual manipulation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Policy type
    parser.add_argument(
        '--policy',
        type=str,
        required=True,
        choices=['flow_matching', 'diffusion', 'act'],
        help='Type of high-level policy to train'
    )

    # Configuration
    parser.add_argument(
        '--config',
        type=str,
        default='scripts/configs/hl_policy_config.yaml',
        help='Path to configuration file'
    )

    # Dataset
    parser.add_argument(
        '--dataset',
        type=str,
        default=None,
        help='Path to demonstration dataset'
    )

    # Training parameters
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Number of training epochs (overrides config)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=None,
        help='Batch size (overrides config)'
    )
    
    parser.add_argument(
        '--no_wrench',
        action='store_true',
        help='Disable external wrench input'
    )

    # Device
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cpu', 'cuda'],
        help='Device for computation'
    )

    # Checkpointing
    parser.add_argument(
        '--checkpoint_dir',
        type=str,
        default=None,
        help='Directory to save checkpoints'
    )
    parser.add_argument(
        '--pred_horizon',
        type=int,
        default=None,
        help='Prediction horizon (chunk size)'
    )
    parser.add_argument(
        '--obs_horizon',
        type=int,
        default=None,
        help='Observation horizon'
    )
    parser.add_argument(
        '--resume_from',
        type=str,
        default=None,
        help='Path to checkpoint to resume training from'
    )

    args = parser.parse_args()

    # Update config with args if needed (hacky but works for now)
    # Ideally we should pass these to train() and merge there
    # For now, we'll rely on config file mostly, but let's handle the flags
    
    # We need to modify the config dict inside train(), but train() loads it.
    # Let's just pass them as args if we were refactoring fully.
    # For now, let's assume the user modifies the config file or we add support later.
    # Actually, let's just update the config dict after loading in train() if we passed it.
    # But train() doesn't take config dict.
    
    # Let's stick to the requested changes.
    # The user asked to "allow setting obs_horizon... default 1".
    # I've added it to the config loading logic in create_policy/load_dataset via config dict.
    # If I want to support CLI override, I need to pass it down.
    
    # Let's update train() signature to accept overrides.
    
    train(
        policy_type=args.policy,
        config_path=args.config,
        dataset_path=args.dataset,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
        resume_from=args.resume_from,
        obs_horizon=args.obs_horizon,
        pred_horizon=args.pred_horizon,
        use_external_wrench=not args.no_wrench if args.no_wrench else None
    )


if __name__ == '__main__':
    main()
