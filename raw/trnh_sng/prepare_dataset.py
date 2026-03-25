import re
import csv
import hashlib
from pathlib import Path

ROOT = Path("raw")
OUT = Path("metadata.csv")

def slugify(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^a-z0-9_]", "", text)
    return text or "unknown"

def sha256_file(path: Path, chunk_size=1024 * 1024):
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def main():
    rows = []
    sample_idx = 1

    person_dirs = [p for p in ROOT.iterdir() if p.is_dir()]
    person_dirs.sort(key=lambda p: p.name.lower())

    for person_dir in person_dirs:
        person_name = person_dir.name
        person_id = slugify(person_name)

        videos = [p for p in person_dir.glob("*.mp4")]
        videos.sort(key=lambda p: p.name.lower())

        # rename folder to normalized person_id (optional)
        normalized_dir = ROOT / person_id
        if person_dir != normalized_dir:
            if normalized_dir.exists():
                # merge mode: keep existing, move files later
                pass
            else:
                person_dir.rename(normalized_dir)
                person_dir = normalized_dir
                videos = [p for p in person_dir.glob("*.mp4")]
                videos.sort(key=lambda p: p.name.lower())

        # rename videos: 0001.mp4, 0002.mp4...
        for i, old_path in enumerate(videos, start=1):
            new_name = f"{i:04d}.mp4"
            new_path = person_dir / new_name
            if old_path != new_path:
                if not new_path.exists():
                    old_path.rename(new_path)
                else:
                    # tránh đè file nếu trùng tên
                    alt_path = person_dir / f"{i:04d}_{sample_idx:06d}.mp4"
                    old_path.rename(alt_path)
                    new_path = alt_path

            rel_path = new_path.as_posix()
            rows.append({
                "sample_id": f"{sample_idx:06d}",
                "file_path": rel_path,
                "person_id": person_id,
                "person_name": person_name,
                "split": "train",  # đổi sau
                "source_ref": "manual_collect",
                "source_url": "",
                "license": "UNKNOWN",
                "rights_note": "No clear redistribution proof yet",
                "sha256": sha256_file(new_path)
            })
            sample_idx += 1

    with OUT.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "sample_id","file_path","person_id","person_name","split",
            "source_ref","source_url","license","rights_note","sha256"
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"Done. Wrote {len(rows)} rows to {OUT}")

if __name__ == "__main__":
    main()
