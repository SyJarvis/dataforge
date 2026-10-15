def dedup_records(records: list[dict]) -> list[dict]:
    newest = {}
    for record in records:
        key = record["id"]
        if key not in newest or record["timestamp"] > newest[key]["timestamp"]:
            newest[key] = record
    return [newest[key] for key in sorted(newest)]
