def dedup_records(records: list[dict]) -> list[dict]:
    newest = {}
    for record in records:
        newest[record["id"]] = record
    return [newest[key] for key in sorted(newest)]
