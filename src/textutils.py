def kmp_search(pattern, text, start_index=0) -> int:
    M = len(pattern)
    N = len(text)

    lps = [0] * M
    j = 0

    compute_lps(pattern, M, lps)

    i = start_index
    while (N - i) >= (M - j):
        if pattern[j] == text[i]:
            i += 1
            j += 1
        if j == M:
            return i - j
        elif i < N and pattern[j] != text[i]:
            if j != 0:
                j = lps[j - 1]
            else:
                i += 1
    return -1

def compute_lps(pattern, M, lps):
    len = 0
    lps[0] = 0
    i = 1

    while i < M:
        if pattern[i] == pattern[len]:
            len += 1
            lps[i] = len
            i += 1
        else:
            if len != 0:
                len = lps[len-1]
            else:
                lps[i] = 0
                i += 1