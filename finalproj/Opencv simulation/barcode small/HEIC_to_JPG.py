from pathlib import Path
from PIL import Image
import pillow_heif
# code to convert from HEIC to JPG
pillow_heif.register_heif_opener()

BASE = Path(__file__).resolve().parent
DATASET = BASE / "dataset"
FOLDERS = [DATASET / "images" / "train", DATASET / "images" / "val"]
HEIC_EXTS = {".heic", ".heif"}

LOG_FILE = BASE / "converted_log.txt"

def log(msg: str):
    print(msg)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")

def convert_and_replace(folder: Path, quality: int = 95, overwrite_jpg: bool = True):
    if not folder.exists():
        log(f"[WARN] folder not found: {folder}")
        return

    converted = deleted = skipped = failed = 0

    heic_files = [p for p in folder.rglob("*")
                  if p.is_file() and p.suffix.lower() in HEIC_EXTS]

    log(f"\n=== {folder} ===")
    log(f"Found HEIC/HEIF: {len(heic_files)}")

    for p in heic_files:
        jpg_path = p.with_suffix(".jpg")

        if jpg_path.exists() and not overwrite_jpg:
            skipped += 1
            log(f"[SKIP] {p.name} -> JPG exists")
            continue

        try:
            with Image.open(p) as im:
                im = im.convert("RGB")
                im.save(jpg_path, format="JPEG", quality=quality, optimize=True)

            if not jpg_path.exists():
                raise RuntimeError("JPG was not created (unknown reason)")

            # try delete original
            p.unlink()
            converted += 1
            deleted += 1
            log(f"[OK] {p.name} -> {jpg_path.name}  (deleted HEIC)")

        except Exception as e:
            failed += 1
            log(f"[FAIL] {p.name}: {e}")

    log(f"Summary: converted={converted}, deleted={deleted}, skipped={skipped}, failed={failed}")

def main():
    # reset log
    if LOG_FILE.exists():
        LOG_FILE.unlink()

    log(f"BASE: {BASE}")
    log(f"DATASET: {DATASET}")

    for folder in FOLDERS:
        convert_and_replace(folder, quality=95, overwrite_jpg=True)

    log("\nDONE. Check converted_log.txt for details.")

if __name__ == "__main__":
    main()



