from __future__ import annotations

import argparse
from pathlib import Path

from tqdm import tqdm

from config_utils import parse_args_with_config
from womd import BevConfig, find_tfrecords, iter_scenarios, save_shard, scenario_to_sample, write_metadata


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--womd_root", default="data/raw/womd")
    parser.add_argument("--output_dir", default="data/womd")
    parser.add_argument("--split", default="training", choices=["training", "validation", "testing"])
    parser.add_argument("--max_files", type=int, default=0)
    parser.add_argument("--max_scenarios", type=int, default=0)
    parser.add_argument("--samples_per_shard", type=int, default=128)
    parser.add_argument("--history_steps", type=int, default=10)
    parser.add_argument("--future_steps", type=int, default=80)
    parser.add_argument("--future_stride", type=int, default=5)
    parser.add_argument("--bev_range", type=float, default=80.0)
    parser.add_argument("--resolution", type=float, default=0.5)
    return parse_args_with_config(parser)


def main():
    args = parse_args()
    cfg = BevConfig(
        x_min=-args.bev_range,
        x_max=args.bev_range,
        y_min=-args.bev_range,
        y_max=args.bev_range,
        resolution=args.resolution,
        history_steps=args.history_steps,
        future_steps=args.future_steps,
        future_stride=args.future_stride,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    shards = find_tfrecords(Path(args.womd_root), args.split, args.max_files)

    samples = []
    total = 0
    shard_id = 0
    pbar = tqdm(iter_scenarios(shards), desc=f"prepare {args.split}")
    for scenario in pbar:
        sample = scenario_to_sample(scenario, cfg)
        if sample is None:
            continue
        samples.append(sample)
        total += 1
        pbar.set_postfix(samples=total)
        if len(samples) >= args.samples_per_shard:
            save_shard(samples, output_dir / f"{args.split}_{shard_id:06d}.pt")
            samples.clear()
            shard_id += 1
        if args.max_scenarios and total >= args.max_scenarios:
            break
    if samples:
        save_shard(samples, output_dir / f"{args.split}_{shard_id:06d}.pt")
    write_metadata(output_dir, cfg, args.split, total)
    print(f"saved {total} samples to {output_dir}")


if __name__ == "__main__":
    main()
