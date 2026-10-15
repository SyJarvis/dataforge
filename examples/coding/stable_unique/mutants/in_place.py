def stable_unique(nums: list[int]) -> list[int]:
    nums[:] = list(dict.fromkeys(nums))
    return nums
