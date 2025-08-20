import os
import random
import argparse
from typing import List


def list_images(root: str) -> List[str]:
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
    p = os.path.abspath(root)
    if not os.path.isdir(p):
        return []
    return [os.path.join(p, f) for f in os.listdir(p) if os.path.splitext(f)[1].lower() in exts]


def write_list(paths: List[str], out_file: str):
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, 'w', encoding='utf-8') as f:
        for p in paths:
            f.write(os.path.abspath(p).replace('\\', '/') + '\n')


def main():
    ap = argparse.ArgumentParser(description='Create subset txt lists for YOLO data.yaml')
    ap.add_argument('--train', default='train/images', help='Path to train/images folder')
    ap.add_argument('--val', default='valid/images', help='Path to valid/images folder (kept full by default)')
    ap.add_argument('--test', default='test/images', help='Path to test/images folder (kept full by default)')
    ap.add_argument('--fraction', type=float, default=0.25, help='Fraction of training images to keep (ignored if --shards is used)')
    ap.add_argument('--shards', type=int, default=0, help='Split the training set into this many equal shards (e.g., 20 for twentieths)')
    ap.add_argument('--shard-index', type=int, default=1, help='1-based index of shard to select when using --shards')
    ap.add_argument('--seed', type=int, default=42, help='Random seed')
    ap.add_argument('--subset-val', action='store_true', help='Also subset validation set')
    ap.add_argument('--subset-test', action='store_true', help='Also subset test set')
    ap.add_argument('--outdir', default='splits', help='Directory to write txt lists')
    args = ap.parse_args()

    random.seed(args.seed)

    train_imgs = list_images(args.train)
    if not train_imgs:
        raise SystemExit(f"No images found in {args.train}")

    # Determine training subset either by random fraction or deterministic shard
    subset_train: List[str]
    if args.shards and args.shards > 0:
        n = len(train_imgs)
        if not (1 <= args.shard_index <= args.shards):
            raise SystemExit(f"--shard-index must be in [1, {args.shards}]")
        # Deterministic shuffle then take the shard slice
        shuffled = train_imgs[:]
        random.shuffle(shuffled)
        base = n // args.shards
        rem = n % args.shards
        # Start index distributes remainder across first 'rem' shards
        i = args.shard_index - 1
        start = i * base + min(i, rem)
        length = base + (1 if i < rem else 0)
        end = start + length
        subset_train = shuffled[start:end]
        frac_pct = int(100 / args.shards)
        train_basename = f"train_{frac_pct}_part{args.shard_index}.txt"
    else:
        k = max(1, int(len(train_imgs) * args.fraction))
        subset_train = random.sample(train_imgs, k)
        train_basename = f'train_{int(args.fraction*100)}.txt'

    val_imgs = list_images(args.val)
    test_imgs = list_images(args.test)

    if args.subset_val and val_imgs:
        kv = max(1, int(len(val_imgs) * args.fraction))
        subset_val = random.sample(val_imgs, kv)
    else:
        subset_val = val_imgs

    if args.subset_test and test_imgs:
        kt = max(1, int(len(test_imgs) * args.fraction))
        subset_test = random.sample(test_imgs, kt)
    else:
        subset_test = test_imgs

    os.makedirs(args.outdir, exist_ok=True)
    train_txt = os.path.join(args.outdir, train_basename)
    val_txt = os.path.join(args.outdir, 'val_full.txt' if not args.subset_val else f'val_{int(args.fraction*100)}.txt')
    test_txt = os.path.join(args.outdir, 'test_full.txt' if not args.subset_test else f'test_{int(args.fraction*100)}.txt')

    write_list(subset_train, train_txt)
    if subset_val:
        write_list(subset_val, val_txt)
    if subset_test:
        write_list(subset_test, test_txt)

    print('Wrote:')
    print(' -', train_txt, len(subset_train))
    print(' -', val_txt, len(subset_val))
    print(' -', test_txt, len(subset_test))


if __name__ == '__main__':
    main()
