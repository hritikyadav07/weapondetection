import os
import random
import argparse


def list_images(root: str):
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
    p = os.path.abspath(root)
    if not os.path.isdir(p):
        return []
    return [os.path.join(p, f) for f in os.listdir(p) if os.path.splitext(f)[1].lower() in exts]


def write_list(paths, out_file):
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, 'w', encoding='utf-8') as f:
        for p in paths:
            f.write(os.path.abspath(p).replace('\\', '/') + '\n')


def main():
    ap = argparse.ArgumentParser(description='Create subset txt lists for YOLO data.yaml')
    ap.add_argument('--train', default='train/images', help='Path to train/images folder')
    ap.add_argument('--val', default='valid/images', help='Path to valid/images folder (kept full by default)')
    ap.add_argument('--test', default='test/images', help='Path to test/images folder (kept full by default)')
    ap.add_argument('--fraction', type=float, default=0.25, help='Fraction of training images to keep')
    ap.add_argument('--seed', type=int, default=42, help='Random seed')
    ap.add_argument('--subset-val', action='store_true', help='Also subset validation set')
    ap.add_argument('--subset-test', action='store_true', help='Also subset test set')
    ap.add_argument('--outdir', default='splits', help='Directory to write txt lists')
    args = ap.parse_args()

    random.seed(args.seed)

    train_imgs = list_images(args.train)
    if not train_imgs:
        raise SystemExit(f"No images found in {args.train}")

    k = max(1, int(len(train_imgs) * args.fraction))
    subset_train = random.sample(train_imgs, k)

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
    train_txt = os.path.join(args.outdir, f'train_{int(args.fraction*100)}.txt')
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
