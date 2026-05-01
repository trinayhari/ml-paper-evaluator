import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Reassemble a split dataset archive from committed part files.")
    parser.add_argument("--out", required=True, help="Output archive path, e.g. data/processed/papers.jsonl.gz")
    args = parser.parse_args()

    out_path = Path(args.out)
    parts = sorted(out_path.parent.glob(out_path.name + ".part-*"))
    if not parts:
        raise FileNotFoundError(f"No parts found for {out_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as out_file:
        for part in parts:
            with open(part, "rb") as part_file:
                out_file.write(part_file.read())
    print(f"restored {out_path} from {len(parts)} parts")


if __name__ == "__main__":
    main()
