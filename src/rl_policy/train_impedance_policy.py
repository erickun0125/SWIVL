"""
DEPRECATED: Use scripts/training/train_ll_policy.py instead.

This module is kept for backward compatibility only.
All training should use the centralized rl_config.yaml configuration.

Usage:
    python scripts/training/train_ll_policy.py --config scripts/configs/rl_config.yaml

To migrate:
1. Copy any custom settings to scripts/configs/rl_config.yaml
2. Use scripts/training/train_ll_policy.py for training
"""

import warnings
import sys
import os

warnings.warn(
    "train_impedance_policy.py is DEPRECATED. "
    "Use scripts/training/train_ll_policy.py with rl_config.yaml instead.",
    DeprecationWarning,
    stacklevel=2
)


def main():
    """Print deprecation notice and exit."""
    print("=" * 80)
    print("DEPRECATED: train_impedance_policy.py")
    print("=" * 80)
    print()
    print("This module has been deprecated. Please use the new unified training script:")
    print()
    print("  python scripts/training/train_ll_policy.py --config scripts/configs/rl_config.yaml")
    print()
    print("All configuration should be done in scripts/configs/rl_config.yaml")
    print("(single source of truth for all RL settings).")
    print()
    print("To migrate:")
    print("  1. Edit scripts/configs/rl_config.yaml with your desired settings")
    print("  2. Run: python scripts/training/train_ll_policy.py")
    print()
    print("=" * 80)
    sys.exit(1)


if __name__ == '__main__':
    main()
