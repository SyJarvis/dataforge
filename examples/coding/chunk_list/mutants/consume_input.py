def chunk_list(items: list[int], k: int) -> list[list[int]]:
    if k <= 0:
        raise ValueError("k must be positive")
    result = []
    while items:
        result.append(items[:k])
        del items[:k]
    return result
