from pathlib import Path

from datasets import load_dataset

OUT = Path("data/raw/tickets.parquet")


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("Tobi-Bueck/customer-support-tickets", split="train")
    df = ds.to_pandas()
    df.to_parquet(OUT, index=False)
    print(f"Saved {len(df)} rows to {OUT}")
    print("Columns:", list(df.columns))


if __name__ == "__main__":
    main()