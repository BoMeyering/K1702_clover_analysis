"""
src/cli.py
`k1702` command line entry point (installed by uv from [project.scripts] in pyproject.toml).

    uv run k1702 train -c configs/flexmatch_sainfoin.yaml [--gpus N | --devices 0,1] [key=value ...]
    uv run k1702 predict CHECKPOINT INPUT [INPUT ...] -o outputs/predictions [--tta]

`train` launches torchrun (torch.distributed.run --standalone) itself when more than one GPU is used;
with one GPU, on CPU, or when already started by torchrun it trains in this process.
BoMeyering 2025
"""

import argparse
import logging
import os
import shlex
import sys


def _add_train(sub):
    p = sub.add_parser(
        "train",
        help="Train a segmentation model (FlexMatch or supervised) from a YAML config",
        description="Train with src/ssl. Any config value can be overridden with dotted key=value "
        "arguments, e.g. training.epochs=5 wandb.mode=disabled.",
    )
    p.add_argument(
        "-c", "--config", required=True, help="Training config YAML (see configs/)"
    )
    p.add_argument(
        "overrides", nargs="*", help="Dotted config overrides, e.g. training.epochs=5"
    )
    g = p.add_mutually_exclusive_group()
    g.add_argument(
        "--gpus",
        type=int,
        default=None,
        help="Number of GPUs / processes (default: all visible GPUs)",
    )
    g.add_argument(
        "--devices",
        default=None,
        help="Comma-separated GPU ids to use, e.g. 0,1 (sets CUDA_VISIBLE_DEVICES)",
    )
    p.add_argument(
        "--backend",
        default=None,
        help="torch.distributed backend (default: nccl on GPU, gloo on CPU)",
    )
    p.add_argument(
        "--master-port",
        type=int,
        default=None,
        help="Rendezvous port for multi-GPU runs (default: torchrun --standalone picks a free one)",
    )
    p.add_argument(
        "--dry-run", action="store_true", help="Print the launch command and exit"
    )
    p.set_defaults(func=cmd_train)


def _add_predict(sub):
    p = sub.add_parser(
        "predict",
        help="Segment images with a trained checkpoint; outputs at full input resolution",
        description="Writes masks/<stem>.png (class indices), color/<stem>.png, overlay/<stem>.jpg and "
        "class_areas.csv under --output-dir, all at the original image resolution.",
    )
    p.add_argument(
        "checkpoint",
        help=".pth checkpoint written by k1702 train (checkpoints/ssl/<run>/...)",
    )
    p.add_argument("inputs", nargs="+", help="Image files and/or directories")
    p.add_argument(
        "-o",
        "--output-dir",
        default="outputs/predictions",
        help="Output directory (default: %(default)s)",
    )
    p.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="Search input directories recursively",
    )
    p.add_argument(
        "--weights",
        choices=["ema", "model"],
        default="ema",
        help="EMA weights (what validation scored, default) or the raw model weights",
    )
    p.add_argument(
        "--save",
        default="mask,color,overlay",
        help="Comma-separated outputs from mask,color,overlay (default: %(default)s)",
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Overlay opacity (default: %(default)s)",
    )
    p.add_argument(
        "--overlay-skip",
        type=int,
        nargs="*",
        default=[0],
        help="Class indices left unpainted in the overlay (default: 0, background)",
    )
    p.add_argument(
        "--tta",
        action="store_true",
        help="Average predictions over horizontal and vertical flips",
    )
    p.add_argument(
        "--resize",
        type=int,
        nargs=2,
        metavar=("H", "W"),
        default=None,
        help="Model input size (default: images.resize from the checkpoint config)",
    )
    p.add_argument(
        "--device",
        default=None,
        help="cuda, cuda:1, cpu ... (default: cuda if available)",
    )
    p.add_argument(
        "--no-amp", dest="amp", action="store_false", help="Disable bf16 autocast"
    )
    p.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=2,
        help="Images per forward pass (default: %(default)s)",
    )
    p.add_argument(
        "-j",
        "--workers",
        type=int,
        default=4,
        help="Data loading workers (default: %(default)s)",
    )
    p.add_argument(
        "--jpeg-quality",
        type=int,
        default=95,
        help="Overlay JPEG quality (default: %(default)s)",
    )
    p.set_defaults(func=cmd_predict)


def cmd_train(args):
    if args.devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.devices

    under_torchrun = "RANK" in os.environ and "LOCAL_WORLD_SIZE" in os.environ
    if under_torchrun:
        nproc = int(os.environ["LOCAL_WORLD_SIZE"])
    else:
        import torch

        available = torch.cuda.device_count()
        nproc = (
            args.gpus
            if args.gpus is not None
            else (len(args.devices.split(",")) if args.devices else max(available, 1))
        )
        if available and nproc > available:
            sys.exit(f"--gpus {nproc} requested but only {available} GPU(s) visible")
    backend = args.backend or ("nccl" if _cuda_available() else "gloo")

    if under_torchrun or nproc <= 1:
        if args.dry_run:
            print(
                f"[k1702] would train in-process: config={args.config} overrides={args.overrides} backend={backend}"
            )
            return 0
        from src.ssl.train import run

        run(args.config, args.overrides, backend)
        return 0

    # Re-launch this CLI under torchrun, one process per GPU
    cmd = [sys.executable, "-m", "torch.distributed.run", f"--nproc_per_node={nproc}"]
    cmd += (
        [f"--master_port={args.master_port}", "--nnodes=1"]
        if args.master_port
        else ["--standalone"]
    )
    cmd += [
        "-m",
        "src.cli",
        "train",
        "-c",
        args.config,
        "--backend",
        backend,
        *args.overrides,
    ]
    print("[k1702] " + shlex.join(cmd[1:]), file=sys.stderr, flush=True)
    if args.dry_run:
        return 0
    os.execv(sys.executable, cmd)


def _cuda_available() -> bool:
    import torch

    return torch.cuda.is_available()


def cmd_predict(args):
    kinds = [k.strip() for k in args.save.split(",") if k.strip()]
    from src.predict import OUTPUT_KINDS, run_predict

    bad = set(kinds) - set(OUTPUT_KINDS)
    if bad:
        sys.exit(
            f"--save: unknown output(s) {sorted(bad)}; choose from {','.join(OUTPUT_KINDS)}"
        )
    run_predict(
        checkpoint=args.checkpoint,
        inputs=args.inputs,
        output_dir=args.output_dir,
        weights=args.weights,
        device=args.device,
        save=kinds,
        alpha=args.alpha,
        overlay_skip=args.overlay_skip,
        tta=args.tta,
        amp=args.amp,
        batch_size=args.batch_size,
        num_workers=args.workers,
        recursive=args.recursive,
        jpeg_quality=args.jpeg_quality,
        resize=tuple(args.resize) if args.resize else None,
    )
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="k1702", description="K1702 segmentation: training and prediction."
    )
    sub = parser.add_subparsers(dest="command", required=True)
    _add_train(sub)
    _add_predict(sub)
    args = parser.parse_args(argv)
    if args.command == "predict":
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(message)s",
            datefmt="%H:%M:%S",
        )
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
