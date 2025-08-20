import os
import sys
from pathlib import Path

VALID_CLASSES = set(range(4))  # adjust if your nc changes


def image_to_label(img_path: Path) -> Path:
    # Map .../images/*.jpg -> .../labels/*.txt
    p = Path(img_path)
    if 'images' not in p.parts:
        return Path()
    parts = list(p.parts)
    parts[parts.index('images')] = 'labels'
    return Path(*parts).with_suffix('.txt')


def parse_line(line: str):
    try:
        parts = line.strip().split()
        if len(parts) != 5:
            return None
        cls = int(float(parts[0]))
        x, y, w, h = map(float, parts[1:])
        return cls, x, y, w, h
    except Exception:
        return None


def is_valid(cls, x, y, w, h):
    if cls not in VALID_CLASSES:
        return False
    # normalized constraints
    if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
        return False
    if not (0.0 < w <= 1.0 and 0.0 < h <= 1.0):
        return False
    # bbox should be inside image after conversion
    if x - w / 2 < 0 or y - h / 2 < 0 or x + w / 2 > 1 or y + h / 2 > 1:
        # allow slight numeric drift but reject grossly invalid
        eps = 1e-6
        return (x - w / 2) >= -eps and (y - h / 2) >= -eps and (x + w / 2) <= 1 + eps and (y + h / 2) <= 1 + eps
    return True


def clean_label_file(label_path: Path) -> tuple[int, int]:
    if not label_path.exists():
        return 0, 0
    kept = []
    total = 0
    with open(label_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if not line.strip():
                continue
            total += 1
            parsed = parse_line(line)
            if not parsed:
                continue
            if is_valid(*parsed):
                kept.append(f"{int(parsed[0])} {parsed[1]:.6f} {parsed[2]:.6f} {parsed[3]:.6f} {parsed[4]:.6f}\n")
    if total and not kept:
        # leave an empty file to indicate no objects; YOLO handles this
        open(label_path, 'w').close()
    elif kept:
        with open(label_path, 'w', encoding='utf-8') as f:
            f.writelines(kept)
    return total, len(kept)


def clean_from_split(split_txt: Path) -> tuple[int, int, int]:
    count_files = 0
    count_boxes_before = 0
    count_boxes_after = 0
    with open(split_txt, 'r', encoding='utf-8') as f:
        for line in f:
            img = Path(line.strip().replace('\\', '/'))
            if not img:
                continue
            label = image_to_label(img)
            if not label:
                continue
            before, after = clean_label_file(label)
            if before:
                count_files += 1
                count_boxes_before += before
                count_boxes_after += after
    return count_files, count_boxes_before, count_boxes_after


if __name__ == '__main__':
    # Defaults to your 5% subset lists
    root = Path(__file__).parent
    train_split = root / 'splits' / 'train_5.txt'
    val_split = root / 'splits' / 'val_full.txt'

    total_files = total_before = total_after = 0
    for sp in (train_split, val_split):
        if sp.exists():
            files, before, after = clean_from_split(sp)
            print(f"Processed {files} label files from {sp.name}: {before} -> {after} boxes")
            total_files += files
            total_before += before
            total_after += after
        else:
            print(f"Split file not found: {sp}")
    print(f"TOTAL files: {total_files}, boxes: {total_before} -> {total_after}")
