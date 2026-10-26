def stable_unique(nums: list[int]) -> list[int]:
    seen = set()
    result = []
    for number in nums:
        if number not in seen:
            seen.add(number)
            result.append(number)
    return result
