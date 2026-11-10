"""
train_flexmatch.py
Thin wrapper around src/ssl/train.py, kept so existing torchrun commands keep working.
Prefer the CLI, which launches torchrun itself:
    uv run k1702 train -c configs/flexmatch_sainfoin.yaml
Direct use:
    uv run torchrun --nproc_per_node=2 train_flexmatch.py -c configs/flexmatch_hq10.yaml
    uv run train_flexmatch.py -c configs/flexmatch_hq10.yaml training.epochs=1 wandb.mode=disabled
BoMeyering 2025
"""

from argparse import ArgumentParser

from src.ssl.train import run

if __name__ == '__main__':
    parser = ArgumentParser(prog='train_flexmatch.py', description='FlexMatch training on the HQ Labelbox masks.')
    parser.add_argument('-c', '--config', default='configs/flexmatch_hq10.yaml', help='Training config YAML')
    parser.add_argument('-b', '--backend', default='nccl', help='torch.distributed backend (nccl or gloo)')
    parser.add_argument('overrides', nargs='*', help='Dotted config overrides, e.g. training.epochs=5')
    args = parser.parse_args()
    run(args.config, args.overrides, args.backend)
