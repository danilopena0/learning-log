import marimo

__generated_with = "0.22.0"
app = marimo.App(width="medium")


@app.cell
def header(mo):
    mo.md("""
    # Arrays & Hash Maps

    | Field   | Value                  |
    |---------|------------------------|
    | Date    | 2026-03-30             |
    | Track   | DSA                    |
    | Time    | 60 min                 |
    | Pattern | Arrays & Hash Maps     |

    **Problems covered:** Two Sum (LC #1) · Valid Anagram (LC #242)
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Concept Overview

    A **hash map** gives O(1) average-case lookups, inserts, and deletes — the key insight
    is that we can trade O(n) space to eliminate a nested loop and drop from O(n²) to O(n).
    The three recurring use-cases are: **complement/pair finding** (store what you've seen,
    check if the needed partner exists), **frequency counting** (tally characters or values),
    and **grouping** (bucket items by a computed key).  Reach for a hash map any time you
    catch yourself writing `for … for` just to answer "have I seen X before?" or "how many
    times does X appear?"
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ---
    ## Problem 1 · Two Sum — LeetCode #1

    > Given an array of integers `nums` and an integer `target`, return the **indices** of
    > the two numbers that add up to `target`. You may assume exactly one solution exists
    > and you may not use the same element twice.

    **Constraints**
    - `2 ≤ len(nums) ≤ 10⁴`
    - `-10⁹ ≤ nums[i] ≤ 10⁹`
    - Exactly one valid answer exists.

    **Example**
    ```
    Input:  nums = [2, 7, 11, 15], target = 9
    Output: [0, 1]   # nums[0] + nums[1] == 9
    ```
    """)
    return


@app.function
def two_sum_brute_force(nums: list[int], target: int) -> list[int]:
    """
    Brute-force approach — check every pair.
    Time:  O(n²)  — two nested loops over n elements
    Space: O(1)   — no extra data structures
    """
    n = len(nums)
    for i in range(n):
        for j in range(i + 1, n):          # start at i+1 to avoid same element
            if nums[i] + nums[j] == target:
                return [i, j]               # found the pair
    return []  # problem guarantees a solution, so we never reach here


@app.function
def two_sum(nums: list[int], target: int) -> list[int]:
    """
    Optimized hash-map approach — one pass.
    Time:  O(n)  — single scan through the array
    Space: O(n)  — hash map stores up to n elements

    Key insight: for each number x, its complement is (target - x).
    If that complement is already in our map, we found the pair.
    Otherwise, store x → index so future elements can find it.
    """
    seen: dict[int, int] = {}  # value → index

    for i, num in enumerate(nums):
        complement = target - num

        if complement in seen:
            # Complement was stored earlier — return both indices
            return [seen[complement], i]

        # Haven't found the pair yet; record this number for later
        seen[num] = i

    return []  # unreachable given problem constraints


@app.cell
def _():
    # --- Two Sum test suite ---

    # Basic example from the problem statement
    assert two_sum([2, 7, 11, 15], 9) == [0, 1]
    assert two_sum_brute_force([2, 7, 11, 15], 9) == [0, 1]

    # Target in the middle of the array
    assert two_sum([3, 2, 4], 6) == [1, 2]
    assert two_sum_brute_force([3, 2, 4], 6) == [1, 2]

    # Duplicate values — same number used at different indices
    assert two_sum([3, 3], 6) == [0, 1]
    assert two_sum_brute_force([3, 3], 6) == [0, 1]

    # Negative numbers
    assert two_sum([-3, 4, 3, 90], 0) == [0, 2]
    assert two_sum_brute_force([-3, 4, 3, 90], 0) == [0, 2]

    # Large negative target
    assert two_sum([-1, -2, -3, -4, -5], -8) == [2, 4]
    assert two_sum_brute_force([-1, -2, -3, -4, -5], -8) == [2, 4]

    # Answer is at the very end
    assert two_sum([1, 2, 3, 4, 5], 9) == [3, 4]
    assert two_sum_brute_force([1, 2, 3, 4, 5], 9) == [3, 4]

    print("All Two Sum tests passed!")
    return


@app.cell
def _(mo):
    mo.md("""
    ### Complexity Analysis — Two Sum

    | Approach       | Time   | Space | Notes                                    |
    |----------------|--------|-------|------------------------------------------|
    | Brute force    | O(n²)  | O(1)  | Two nested loops; no extra memory        |
    | Hash map (1-pass) | O(n) | O(n)  | One scan; map holds at most n entries    |

    **Trade-off:** We spend O(n) extra memory to gain O(n) time — almost always the right
    call in an interview. The map never stores more than n − 1 entries because we return
    the moment we find the complement.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ---
    ## Problem 2 · Valid Anagram — LeetCode #242

    > Given two strings `s` and `t`, return `True` if `t` is an anagram of `s`,
    > and `False` otherwise.  An anagram uses all the original letters exactly once.

    **Constraints**
    - `1 ≤ len(s), len(t) ≤ 5 × 10⁴`
    - `s` and `t` consist of lowercase English letters *(for the basic version)*.

    **Example**
    ```
    Input:  s = "anagram", t = "nagaram"   →  True
    Input:  s = "rat",     t = "car"        →  False
    ```
    """)
    return


@app.function
def is_anagram_sort(s: str, t: str) -> bool:
    """
    Approach 1 — Sorting.
    Time:  O(n log n)  — dominated by sort
    Space: O(n)        — sorted() creates new strings

    Two strings are anagrams iff their sorted characters are identical.
    Simple, readable, but slower than the hash-map approaches.
    """
    if len(s) != len(t):
        return False
    return sorted(s) == sorted(t)


@app.cell
def valid_anagram_hashmap():
    from collections import Counter

    def is_anagram_counter(s: str, t: str) -> bool:
        """
        Approach 2 — Hash map / Counter.
        Time:  O(n)  — two linear scans to build counts, one to compare
        Space: O(k)  — k = number of unique characters (≤ 26 for lowercase)

        Counter subtracts cleanly: Counter(s) - Counter(t) is empty iff
        every character in s appears at least as many times in t (and
        lengths match ensures no extra characters in t).
        """
        if len(s) != len(t):
            return False
        return Counter(s) == Counter(t)

    return (is_anagram_counter,)


@app.function
def is_anagram_array(s: str, t: str) -> bool:
    """
    Approach 3 (bonus) — Fixed-size array of 26.
    Time:  O(n)   — two linear scans
    Space: O(1)   — array is always exactly 26 integers

    Works only for lowercase English letters.  Maps each character to
    an index via ord(c) - ord('a'), increments for s, decrements for t.
    If all counts return to zero, it's an anagram.
    """
    if len(s) != len(t):
        return False

    counts = [0] * 26
    for c in s:
        counts[ord(c) - ord('a')] += 1
    for c in t:
        counts[ord(c) - ord('a')] -= 1

    # Any non-zero entry means a character mismatch
    return all(c == 0 for c in counts)


@app.cell
def _(is_anagram_counter):
    # --- Valid Anagram test suite ---

    solutions = [is_anagram_sort, is_anagram_counter, is_anagram_array]

    for is_anagram in solutions:
        # Standard true cases
        assert is_anagram("anagram", "nagaram") is True
        assert is_anagram("listen", "silent") is True
        assert is_anagram("a", "a") is True           # single character

        # Standard false cases
        assert is_anagram("rat", "car") is False
        assert is_anagram("hello", "world") is False

        # Length mismatch — fast-path rejection
        assert is_anagram("ab", "abc") is False
        assert is_anagram("abc", "ab") is False

        # Empty strings — anagram of each other
        assert is_anagram("", "") is True

        # Repeated characters
        assert is_anagram("aab", "baa") is True
        assert is_anagram("aab", "bab") is False

        # All same character
        assert is_anagram("aaaa", "aaaa") is True
        assert is_anagram("aaaa", "aaab") is False

    print("All Valid Anagram tests passed!")
    return


@app.cell
def _(mo):
    mo.md("""
    ### Complexity Analysis — Valid Anagram

    | Approach            | Time       | Space  | Notes                                          |
    |---------------------|------------|--------|------------------------------------------------|
    | Sorting             | O(n log n) | O(n)   | Simple; creates sorted copies                  |
    | Counter / hash map  | O(n)       | O(k)   | k = unique chars; practical best choice        |
    | Array of 26         | O(n)       | O(1)   | Fastest in practice; lowercase-only constraint |

    For the follow-up *"what if inputs contain Unicode?"* — the array approach breaks;
    use `Counter` which handles arbitrary characters without code changes.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ---
    ## Pattern Recognition — Hash Map Cheat Sheet

    | Situation                              | Go-to tool                         |
    |----------------------------------------|------------------------------------|
    | Need O(1) lookup of a value            | `dict` / `set`                     |
    | Count character / element frequency    | `collections.Counter`              |
    | Count with a default value             | `collections.defaultdict(int)`     |
    | Find complement / pair                 | Store seen values as keys          |
    | Group items by a computed key          | `defaultdict(list)`                |

    ### Other LeetCode problems using the same pattern

    | Problem                         | LC # | Key idea                                         |
    |---------------------------------|------|--------------------------------------------------|
    | Group Anagrams                  | 49   | Use sorted string as hash key to group           |
    | Contains Duplicate              | 217  | Set membership check in one pass                 |
    | Intersection of Two Arrays      | 349  | Convert one array to set, scan the other         |
    | Longest Consecutive Sequence    | 128  | Set for O(1) lookup; only start chains at minima |
    | Subarray Sum Equals K           | 560  | Prefix-sum + hash map for complement counts      |
    | Top K Frequent Elements         | 347  | Counter + heap or bucket sort                    |
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ---
    ## Interview Talking Points

    ### Two Sum — 30-second explanation
    > "I'll do a single pass.  For each element I compute its complement
    > `target - num` and check a hash map of values I've already seen.
    > If the complement is there I return both indices; otherwise I store
    > the current value.  One loop, O(n) time, O(n) space."

    ---

    ### Common follow-up questions

    **What if the array is already sorted?**
    Use the **two-pointer** technique instead — left and right pointers converge
    in O(n) time with O(1) space, no hash map needed.

    **What if there are multiple valid pairs?**
    Collect all pairs: don't return early, keep scanning and append each hit.
    Be careful with duplicates — decide whether the same index can appear in
    multiple output pairs (usually not).

    **What if there are duplicate values?**
    The hash map stores the *most recent* index for a value.  If duplicates
    could form a valid pair (e.g. `[3, 3], target=6`) this still works because
    we check the complement *before* writing the current index into the map.

    **What about integer overflow?**
    Python integers are arbitrary precision, so no issue.  In Java/C++ I'd
    cast to `long` before the addition.

    ---

    ### Time–Space trade-off
    > "I'm trading O(n) extra space for a factor-of-n speedup — going from
    > O(n²) brute force to O(n).  In virtually every real-world scenario that
    > trade is worth it; memory is cheap and n can be large.  If memory is
    > genuinely constrained and the array is sorted, I'd switch to two pointers
    > for the O(1)-space solution."
    """)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
