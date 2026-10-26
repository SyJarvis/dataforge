def parse_config(text: str) -> dict[str, str]:
    result = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError("configuration line has no equals sign")
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        if not key:
            raise ValueError("configuration key is empty")
        result[key] = value
    return result
