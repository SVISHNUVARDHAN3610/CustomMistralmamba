import json

input_file, output_file = "training_logs.jsonl", "training_logs_every_100.jsonl"
invalid = "./tmp"
with (
    open(input_file, "r", encoding="utf-8") as infile,
    open(output_file, "w", encoding="utf-8") as outfile,
):
    for line in infile:
        line = line.strip()

        if not line:
            continue

        record = json.loads(line)

        if record["step"] % 100 == 0:
            outfile.write(json.dumps(record) + "\n")

print(f"Filtered logs saved to: {output_file}")
