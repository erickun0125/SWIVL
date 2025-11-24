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


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_policy(policy_type: str, config: Dict[str, Any], device: str):
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

    if policy_type == 'flow_matching':
        policy = FlowMatchingPolicy(device=device)
    elif policy_type == 'diffusion':
        policy = DiffusionPolicy(device=device)
    elif policy_type == 'act':
        policy = ACTPolicy(device=device)
    else:
        raise ValueError(f"Unknown policy type: {policy_type}")

    print(f"✓ Policy created with {sum(p.numel() for p in policy.parameters())} parameters")
    return policy


def create_optimizer(policy, config: Dict[str, Any]):
    """Create optimizer for training."""
    optimizer_type = config.get('hl_training', {}).get('optimizer', 'adam').lower()
    lr = config.get('hl_training', {}).get('learning_rate', 1e-4)
    weight_decay = config.get('hl_training', {}).get('weight_decay', 1e-5)

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
    scheduler_config = config.get('hl_training', {}).get('lr_scheduler', {})
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
        T = 50
        state_dim = 18
        action_dim = 6
        
        obs = np.random.randn(N, T, state_dim).astype(np.float32)
        action = np.random.randn(N, T, action_dim).astype(np.float32)
        
        class DummyDataset(torch.utils.data.Dataset):
            def __init__(self, obs, action):
                self.obs = torch.from_numpy(obs)
                self.action = torch.from_numpy(action)
            def __len__(self): return len(self.obs)
            def __getitem__(self, idx): return self.obs[idx], self.action[idx]
            
        dataset = DummyDataset(obs, action)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        return random_split(dataset, [train_size, val_size])

    import h5py
    
    class BiArtHDF5Dataset(torch.utils.data.Dataset):
        def __init__(self, hdf5_path, obs_horizon=1, pred_horizon=1):
            """
            Args:
                hdf5_path: Path to .h5 file
                obs_horizon: Number of past steps to stack (default 1 for now)
                pred_horizon: Number of future steps to predict (default 1 for now)
            """
            super().__init__()
            self.file_path = hdf5_path
            self.obs_horizon = obs_horizon
            self.pred_horizon = pred_horizon
            
            # Load all data into memory (for simplicity)
            self.episodes = []
            with h5py.File(self.file_path, 'r') as f:
                num_demos = f.attrs.get('num_demos', 0)
                print(f"Found {num_demos} demos in file.")
                
                for i in range(num_demos):
                    demo_key = f'demo_{i}'
                    if demo_key not in f: continue
                    
                    g = f[demo_key]
                    # Load obs
                    ee_poses = g['obs']['ee_poses'][:]  # (T, 2, 3)
                    link_poses = g['obs']['link_poses'][:] # (T, 2, 3)
                    wrenches = g['obs']['external_wrenches'][:] # (T, 2, 3)
                    
                    # Flatten state: (T, 18)
                    T = ee_poses.shape[0]
                    state = np.concatenate([
                        ee_poses.reshape(T, -1), 
                        link_poses.reshape(T, -1), 
                        wrenches.reshape(T, -1)
                    ], axis=1)
                    
                    # Load action: (T, 2, 3) -> (T, 6)
                    action = g['action'][:].reshape(T, -1)
                    
                    self.episodes.append({
                        'state': torch.from_numpy(state.astype(np.float32)),
                        'action': torch.from_numpy(action.astype(np.float32))
                    })
            
            # Create indices
            self.indices = []
            for ep_idx, episode in enumerate(self.episodes):
                seq_len = len(episode['state'])
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
            obs_seq = episode['state'][start:end] # (obs_horizon, state_dim)
            
            # Get action window
            # Assuming policy expects (pred_horizon, action_dim)
            # For simple diffusion/ACT, usually predict future chunk
            act_start = t
            act_end = t + self.pred_horizon
            action_seq = episode['action'][act_start:act_end] # (pred_horizon, action_dim)
            
            return obs_seq, action_seq

    # Determine horizons based on config/policy type
    # For generic compatibility, default to horizons that match the policies
    # FlowMatching: state_dim (18) -> action_dim (6) (usually 1->1 mapping in basic form)
    # But diffusion/ACT use history.
    # We need to look at the config passed in.
    
    # Safe defaults
    obs_h = config.get('hl_training', {}).get('obs_horizon', 1)
    pred_h = config.get('hl_training', {}).get('action_horizon', 1)
    
    # Create dataset
    full_dataset = BiArtHDF5Dataset(dataset_path, obs_horizon=obs_h, pred_horizon=pred_h)
    
    # Split
    val_ratio = config.get('hl_training', {}).get('val_ratio', 0.2)
    val_size = int(val_ratio * len(full_dataset))
    train_size = len(full_dataset) - val_size
    
    return random_split(full_dataset, [train_size, val_size])


def train_epoch(
    policy,
    train_loader,
    optimizer,
    device: str,
    epoch: int,
    writer: Optional[SummaryWriter] = None,
    log_interval: int = 100
):
    """Train for one epoch."""
    policy.train()
    total_loss = 0.0
    num_batches = 0

    for batch_idx, (states, actions) in enumerate(train_loader):
        states = states.to(device)
        actions = actions.to(device)

        # Forward pass
        optimizer.zero_grad()
        loss = policy.compute_loss(states, actions)

        # Backward pass
        loss.backward()
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


def validate(policy, val_loader, device: str):
    """Validate policy."""
    policy.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for states, actions in val_loader:
            states = states.to(device)
            actions = actions.to(device)

            loss = policy.compute_loss(states, actions)
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
    resume_from: Optional[str] = None
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
    print(f"Using device: {device}")

    # Override config with command-line arguments
    if dataset_path is None:
        dataset_path = config.get('hl_training', {}).get('dataset', {}).get('path', './data/demonstrations')
    if num_epochs is None:
        num_epochs = config.get('hl_training', {}).get('num_epochs', 100)
    if batch_size is None:
        batch_size = config.get('hl_training', {}).get('batch_size', 64)
    if checkpoint_dir is None:
        checkpoint_dir = config.get('hl_training', {}).get('output', {}).get('checkpoint_dir', './checkpoints')

    # Create policy
    policy = create_policy(policy_type, config, device)

    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_loss = float('inf')
    if resume_from is not None and os.path.exists(resume_from):
        print(f"Resuming from checkpoint: {resume_from}")
        checkpoint = torch.load(resume_from, map_location=device)
        policy.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        print(f"Resumed from epoch {start_epoch}")

    # Load dataset
    train_dataset, val_dataset = load_dataset(dataset_path, config)

    if train_dataset is None:
        print("\n" + "=" * 80)
        print("ERROR: Dataset loading not implemented!")
        print("=" * 80)
        print("\nTo complete the implementation:")
        print("1. Implement load_dataset() function in this script")
        print("2. Dataset should return (states, actions) pairs")
        print("3. States: Current observation (ee_poses, link_poses, external_wrenches)")
        print("4. Actions: Desired poses for both end-effectors")
        print("\nFor now, training is skipped.")
        print("=" * 80)
        return

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Create optimizer and scheduler
    optimizer = create_optimizer(policy, config)
    scheduler = create_scheduler(optimizer, config, num_epochs)

    # Create tensorboard writer
    log_dir = config.get('hl_training', {}).get('logging', {}).get('tensorboard_log', './logs/hl_policy/')
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
            policy, train_loader, optimizer, device, epoch, writer,
            log_interval=config.get('hl_training', {}).get('logging', {}).get('log_interval', 100)
        )
        print(f"  Train Loss: {train_loss:.6f}")

        # Validate
        val_loss = validate(policy, val_loader, device)
        print(f"  Val Loss: {val_loss:.6f}")

        # Log to tensorboard
        writer.add_scalar('train/epoch_loss', train_loss, epoch)
        writer.add_scalar('val/epoch_loss', val_loss, epoch)
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

        # Update learning rate
        if scheduler is not None:
            scheduler.step()

        # Save checkpoint
        save_freq = config.get('hl_training', {}).get('logging', {}).get('save_freq', 10)
        if (epoch + 1) % save_freq == 0:
            checkpoint_path = os.path.join(
                checkpoint_dir,
                f"{policy_type}_epoch_{epoch+1}.pth"
            )
            torch.save({
                'epoch': epoch,
                'model_state_dict': policy.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'best_val_loss': best_val_loss
            }, checkpoint_path)
            print(f"  Saved checkpoint: {checkpoint_path}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(checkpoint_dir, f"{policy_type}_best.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': policy.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'best_val_loss': best_val_loss
            }, best_model_path)
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
        default='scripts/configs/rl_config.yaml',
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
        '--resume_from',
        type=str,
        default=None,
        help='Path to checkpoint to resume training from'
    )

    args = parser.parse_args()

    # Train
    train(
        policy_type=args.policy,
        config_path=args.config,
        dataset_path=args.dataset,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
        resume_from=args.resume_from
    )


if __name__ == '__main__':
    main()
