def max_false_run(flags: list[bool]) -> int:
    longest = current = 0
    for flag in flags:
        if flag:
            longest = max(longest, current)
            current = 0
        else:
            current += 1
    return longest
