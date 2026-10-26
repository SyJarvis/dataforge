def chunk_list(items: list[int], k: int) -> list[list[int]]:
    if k <= 0:
        raise ValueError("k must be positive")
    return [items[index:index + k] for index in range(0, len(items), k)]
