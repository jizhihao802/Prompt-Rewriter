from datasets import Dataset
import argparse

def halve_parquet(input_path, output_path):
    print(f"🔹 Loading {input_path} ...")
    dataset = Dataset.from_parquet(input_path)

    total_len = len(dataset)
    half_len = total_len // 2
    quarter_len = total_len // 4

    print(f"➡️ Total samples: {total_len}")
    print(f"➡️ Keeping only: {total_len - quarter_len} samples")

    # 只保留前一半
    new_dataset = dataset.select(range(quarter_len,total_len))

    print(f"➡️ New dataset length: {len(new_dataset)}")

    print(f"💾 Saving to {output_path}")
    new_dataset.to_parquet(output_path)
    print("✅ Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)

    args = parser.parse_args()

    halve_parquet(args.input, args.output)
