def max_false_run(flags: list[bool]) -> int:
    longest = current = 0
    for flag in flags:
        if flag:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest
