def binary_search_first(nums: list[int], target: int) -> int:
    left, right = 0, len(nums)
    while left < right:
        middle = (left + right) // 2
        if nums[middle] <= target:
            left = middle + 1
        else:
            right = middle
    index = left - 1
    return index if index >= 0 and nums[index] == target else -1
