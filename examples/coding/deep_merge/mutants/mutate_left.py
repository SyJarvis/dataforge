def deep_merge(left: dict, right: dict) -> dict:
    for key, value in right.items():
        if key in left and isinstance(left[key], dict) and isinstance(value, dict):
            left[key] = deep_merge(left[key], value)
        else:
            left[key] = value
    return left
