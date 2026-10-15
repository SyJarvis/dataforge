def deep_merge(left: dict, right: dict) -> dict:
    result = left.copy()
    result.update(right)
    return result
