import argparse
import os
import random
import shutil
import csv
from pathlib import Path
from collections import defaultdict

def list_classes(src):
    return sorted([p for p in os.listdir(src) if (Path(src)/p).is_dir()])

def list_objects(class_dir):
    # Each object is a subdirectory (e.g., airplane_0001)
    return sorted([p for p in os.listdir(class_dir) if (Path(class_dir)/p).is_dir()])

def ensure_dir(p): 
    Path(p).mkdir(parents=True, exist_ok=True)

def perform_split(src, dst, train_pct=0.8, seed=42, mode='symlink', dry_run=False, csv_out=True):
    random.seed(seed)
    src = Path(src)
    dst = Path(dst) if dst else None

    classes = list_classes(src)
    summary = defaultdict(lambda: {"n_objects":0, "train":0, "test":0})
    mapping = []  # list of tuples (class, object, dest, n_views)

    for cls in classes:
        class_dir = src / cls
        objects = list_objects(class_dir)
        n = len(objects)
        summary[cls]["n_objects"] = n
        if n == 0:
            continue

        random.shuffle(objects)
        n_train = int(round(train_pct * n))

        train_objs = set(objects[:n_train])
        test_objs = set(objects[n_train:])

        summary[cls]["train"] = len(train_objs)
        summary[cls]["test"] = len(test_objs)

        if dry_run:
            # just record mapping info
            for obj in objects:
                dest = 'train' if obj in train_objs else 'test'
                n_views = len(list((class_dir/obj).iterdir()))
                mapping.append((cls, obj, dest, n_views))
            continue

        # create target dirs
        assert dst is not None, "dst must be provided when not dry-run"
        for split_name, objset in (("train", train_objs), ("test", test_objs)):
            for obj in objset:
                src_obj_dir = class_dir / obj
                dst_obj_dir = dst / split_name / cls / obj
                ensure_dir(dst_obj_dir)
                # iterate files and copy/symlink
                for item in src_obj_dir.iterdir():
                    src_path = item
                    dst_path = dst_obj_dir / item.name
                    if mode == 'copy':
                        shutil.copy2(src_path, dst_path)
                    elif mode == 'symlink':
                        # if symlink exists, remove then create
                        if dst_path.exists() or dst_path.is_symlink():
                            dst_path.unlink()
                        os.symlink(os.path.abspath(src_path), dst_path)
                    else:
                        raise ValueError("mode must be 'copy' or 'symlink'")
                mapping.append((cls, obj, split_name, len(list(src_obj_dir.iterdir()))))

    # Write summary and CSV
    total_objects = sum(summary[c]['n_objects'] for c in summary)
    total_train = sum(summary[c]['train'] for c in summary)
    total_test  = sum(summary[c]['test']  for c in summary)

    print(f"Total classes: {len(classes)}")
    print(f"Total objects: {total_objects}")
    print(f"Train objects: {total_train}  Test objects: {total_test}  (train_pct={train_pct})")
    print("Per-class counts (class: total / train / test):")
    for c in sorted(summary):
        v = summary[c]
        print(f"  {c}: {v['n_objects']} / {v['train']} / {v['test']}")

    if csv_out:
        csv_path = (dst / "split.csv") if dst else Path("split_preview.csv")
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["class", "object", "split", "n_views"])
            for row in mapping:
                writer.writerow(row)
        print(f"Wrote mapping CSV to {csv_path}")

    if dry_run:
        print("Dry-run complete. No files were copied or linked.")

    return summary, mapping

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--src', required=True, help='Source dataset directory (ModelNet40-12View)')
    p.add_argument('--dst', required=False, help='Destination directory for split (e.g., ModelNet40-split)')
    p.add_argument('--train-pct', type=float, default=0.8, help='Fraction for train (0..1). Default 0.8')
    p.add_argument('--seed', type=int, default=42, help='Random seed')
    p.add_argument('--mode', choices=['symlink','copy'], default='symlink', help='How to create split data (symlink faster)')
    p.add_argument('--dry-run', action='store_true', help='Only print counts and write preview CSV; do not copy/link files')
    p.add_argument('--no-csv', dest='csv_out', action='store_false', help='Do not write mapping CSV')
    return p.parse_args()

if __name__ == '__main__':
    args = parse_args()
    if args.dry_run and args.dst:
        print("Warning: --dst will be ignored in dry-run mode")
    if args.dry_run or args.dst:
        perform_split(
            src=args.src,
            dst=args.dst,
            train_pct=args.train_pct,
            seed=args.seed,
            mode=args.mode,
            dry_run=args.dry_run,
            csv_out=args.csv_out,
        )
    else:
        raise SystemExit("Error: when not using --dry-run you must pass --dst to specify output directory.")