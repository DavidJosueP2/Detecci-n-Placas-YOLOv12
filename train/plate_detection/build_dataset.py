from pathlib import Path
import hashlib
import shutil

from PIL import Image
import yaml


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent.parent
SOURCE = PROJECT_ROOT / "datasets" / "Placas Ecuador"
DEST = PROJECT_ROOT / "datasets" / "placas_ecuador_preparado"
DEST_YAML = ROOT / "configs" / "placas_ecuador_preparado.yaml"
SPLITS = ("train", "valid", "test")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def file_hash(path):
    return hashlib.md5(path.read_bytes()).hexdigest()


def validate_label(label_path):
    errors = []
    lines = [line.strip() for line in label_path.read_text().splitlines() if line.strip()]
    for idx, line in enumerate(lines, start=1):
        parts = line.split()
        if len(parts) != 5:
            errors.append(f"{label_path.name}:{idx} formato invalido")
            continue

        cls, x, y, w, h = parts
        try:
            cls_id = int(cls)
            values = [float(x), float(y), float(w), float(h)]
        except ValueError:
            errors.append(f"{label_path.name}:{idx} contiene valores no numericos")
            continue

        if cls_id != 0:
            errors.append(f"{label_path.name}:{idx} clase inesperada {cls_id}")

        cx, cy, bw, bh = values
        if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < bw <= 1 and 0 < bh <= 1):
            errors.append(f"{label_path.name}:{idx} bbox fuera de rango")

    return lines, errors


def copy_pair(image_path, label_path, dest_split):
    dest_images = DEST / dest_split / "images"
    dest_labels = DEST / dest_split / "labels"
    dest_images.mkdir(parents=True, exist_ok=True)
    dest_labels.mkdir(parents=True, exist_ok=True)
    shutil.copy2(image_path, dest_images / image_path.name)
    shutil.copy2(label_path, dest_labels / label_path.name)


def prepare_split(split):
    image_dir = SOURCE / split / "images"
    label_dir = SOURCE / split / "labels"
    images = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTS)

    seen_hashes = set()
    stats = {
        "images": 0,
        "labels": 0,
        "empty_labels": 0,
        "duplicates_removed": 0,
        "errors": [],
        "sizes": {},
    }

    for image_path in images:
        label_path = label_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            stats["errors"].append(f"Falta label para {image_path.name}")
            continue

        try:
            with Image.open(image_path) as image:
                stats["sizes"][image.size] = stats["sizes"].get(image.size, 0) + 1
        except Exception as exc:
            stats["errors"].append(f"No se pudo abrir {image_path.name}: {exc}")
            continue

        lines, label_errors = validate_label(label_path)
        stats["errors"].extend(label_errors)
        if not lines:
            stats["empty_labels"] += 1

        digest = file_hash(image_path)
        if split == "train" and digest in seen_hashes:
            stats["duplicates_removed"] += 1
            continue
        seen_hashes.add(digest)

        copy_pair(image_path, label_path, split)
        stats["images"] += 1
        stats["labels"] += 1

    return stats


def write_yaml():
    data = {
        "path": str(DEST),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": 1,
        "names": ["placa"],
    }
    DEST_YAML.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def main():
    if DEST.exists():
        shutil.rmtree(DEST)

    report = {split: prepare_split(split) for split in SPLITS}
    write_yaml()

    print(f"Dataset preparado: {DEST}")
    print(f"YAML generado: {DEST_YAML}")
    for split, stats in report.items():
        print(
            f"{split}: images={stats['images']} labels={stats['labels']} "
            f"empty_labels={stats['empty_labels']} duplicates_removed={stats['duplicates_removed']}"
        )
        print(f"  sizes={stats['sizes']}")
        if stats["errors"]:
            print("  errores:")
            for error in stats["errors"][:10]:
                print(f"  - {error}")

    has_errors = any(stats["errors"] for stats in report.values())
    return 1 if has_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
