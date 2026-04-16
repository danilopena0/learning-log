import marimo

__generated_with = "0.22.0"
app = marimo.App(width="medium")


@app.cell
def header(mo):
    mo.md("""
    # Sliding Window & Two Pointers

    | Field    | Value                              |
    |----------|------------------------------------|
    | Date     | 2026-04-02                         |
    | Track    | DSA                                |
    | Time     | 60 min                             |
    | Patterns | Sliding Window, Two Pointers       |

    **Problems covered:** Best Time to Buy/Sell Stock (LC #121) · Container With Most Water (LC #11)
    """)
    return


@app.cell
def pattern_overview(mo):
    mo.md("""
    ## Pattern Overview

    ### Two Pointers

    **Use when:** working with sorted arrays, or when you need to find pairs/subarrays satisfying a condition.

    **Core idea:** maintain two indices that move toward each other (or in the same direction) to reduce O(n²) to O(n).

    **Two variants:**
    1. **Opposite ends → inward:** start at both ends, converge based on a condition
    2. **Slow/fast (same direction):** both pointers move forward, one ahead of the other

    ---

    ### Sliding Window

    **Use when:** finding optimal subarray/substring of variable or fixed length.

    **Core idea:** maintain a window `[left, right]` that expands and contracts, tracking a running state.

    **Key insight:** instead of recomputing from scratch for each subarray, update incrementally as the window slides.

    > The sliding window is the **same-direction two-pointer variant with state tracking**.

    ---

    ### Quick Decision Tree

    ```
    Need to find a pair in a sorted array?        → opposite-end two pointers
    Need optimal contiguous subarray?             → sliding window
    Need to compare elements from both ends?      → opposite-end two pointers
    Need min/max subarray with a constraint?      → sliding window with expand/shrink
    ```
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ---
    ## Problem 1 · Best Time to Buy and Sell Stock — LeetCode #121

    > You are given an array `prices` where `prices[i]` is the price of a given stock on day `i`.
    > Choose a **single day to buy** and a **different future day to sell** to maximize profit.
    > Return the maximum profit, or `0` if no profit is possible.

    **Constraints**
    - `1 ≤ len(prices) ≤ 10⁵`
    - `0 ≤ prices[i] ≤ 10⁴`

    **Example**
    ```
    Input:  prices = [7, 1, 5, 3, 6, 4]
    Output: 5   # buy at price 1 (day 1), sell at price 6 (day 4)
    ```
    """)
    return


@app.function
def max_profit_brute(prices: list[int]) -> int:
    """
    Brute-force — check every buy/sell pair.
    Time:  O(n²)  — two nested loops over all pairs
    Space: O(1)   — no extra data structures

    Note: "This is what I'd mention first to show I understand the naive approach."
    """
    if len(prices) < 2:
        return 0

    best = 0
    n = len(prices)
    for i in range(n):               # buy day
        for j in range(i + 1, n):   # sell day (must be strictly after buy)
            profit = prices[j] - prices[i]
            best = max(best, profit)
    return best


@app.function
def max_profit(prices: list[int]) -> int:
    """
    Optimized — sliding window / one-pass approach.
    Time:  O(n)  — single left-to-right scan
    Space: O(1)  — only two variables tracked

    Key insight: at each position (right pointer), the best we can do is
    sell here and subtract the cheapest price seen so far (left pointer).
    The left pointer jumps forward whenever we find a new minimum.

    This IS a sliding window: left tracks the best buy candidate, right
    scans forward looking for the best sell. The window implicitly
    represents the best [buy_day, sell_day] pair ending at today.
    """
    if not prices:
        return 0

    min_price_so_far = prices[0]   # left pointer: cheapest buy seen
    best_profit = 0                # running maximum profit

    for price in prices[1:]:       # right pointer scans forward day by day
        # If we sold today, would this beat our current best?
        best_profit = max(best_profit, price - min_price_so_far)
        # Update cheapest buy opportunity seen so far
        min_price_so_far = min(min_price_so_far, price)

    return best_profit


@app.cell
def buy_sell_visualization():
    import matplotlib as _matplotlib
    _matplotlib.use("Agg")
    import matplotlib.pyplot as _plt

    prices = [7, 1, 5, 3, 6, 4]
    days = list(range(len(prices)))

    # Compute min_price_so_far at each day for the dashed tracking line
    running_min = []
    cur_min = prices[0]
    for p in prices:
        cur_min = min(cur_min, p)
        running_min.append(cur_min)

    buy_day, sell_day = 1, 4
    buy_price, sell_price = prices[buy_day], prices[sell_day]

    _fig, _ax = _plt.subplots(figsize=(8, 4))
    _fig.patch.set_facecolor("white")
    _ax.set_facecolor("white")

    # Price line
    _ax.plot(days, prices, color="#2c3e50", linewidth=2.5, marker="o",
            markersize=7, zorder=3, label="Price")

    # Shade profit region between buy and sell days
    _ax.fill_between(
        days[buy_day : sell_day + 1],
        prices[buy_day : sell_day + 1],
        buy_price,
        alpha=0.18, color="#27ae60",
        label=f"Profit region (+{sell_price - buy_price})"
    )

    # Dashed line tracking min_price_so_far (the "left pointer" state)
    _ax.step(days, running_min, where="post", color="#e67e22",
            linestyle="--", linewidth=1.8, zorder=2, label="min_price_so_far")

    # Buy annotation — green arrow
    _ax.annotate(
        f"BUY\n${buy_price}",
        xy=(buy_day, buy_price),
        xytext=(buy_day + 0.4, buy_price + 1.8),
        arrowprops=dict(arrowstyle="->", color="#27ae60", lw=2.0),
        color="#27ae60", fontsize=9, fontweight="bold"
    )

    # Sell annotation — red arrow
    _ax.annotate(
        f"SELL\n${sell_price}",
        xy=(sell_day, sell_price),
        xytext=(sell_day - 1.2, sell_price + 1.2),
        arrowprops=dict(arrowstyle="->", color="#e74c3c", lw=2.0),
        color="#e74c3c", fontsize=9, fontweight="bold"
    )

    _ax.set_xticks(days)
    _ax.set_xticklabels([f"Day {d}" for d in days])
    _ax.set_ylabel("Price ($)")
    _ax.set_title(
        "LC #121 — Best Time to Buy and Sell Stock\n"
        "[7, 1, 5, 3, 6, 4]  →  profit = 5",
        fontsize=11
    )
    _ax.legend(loc="upper right", fontsize=8)
    _ax.grid(axis="y", alpha=0.3)
    _ax.set_ylim(0, 9.5)
    _fig.tight_layout()
    return _fig


@app.cell
def _():
    # --- Best Time to Buy and Sell Stock — test suite ---

    assert max_profit([7, 1, 5, 3, 6, 4]) == 5    # basic: buy at 1, sell at 6
    assert max_profit([7, 6, 4, 3, 1]) == 0         # descending: no profitable trade
    assert max_profit([1, 2]) == 1                   # minimal two-element case
    assert max_profit([2, 4, 1]) == 2                # best buy is at the start
    assert max_profit([3, 3, 3, 3]) == 0             # flat: no profit possible
    assert max_profit([1]) == 0                      # single element: can't sell
    assert max_profit([]) == 0                       # empty: discuss with interviewer

    assert max_profit_brute([7, 1, 5, 3, 6, 4]) == 5
    assert max_profit_brute([7, 6, 4, 3, 1]) == 0
    assert max_profit_brute([1, 2]) == 1
    assert max_profit_brute([2, 4, 1]) == 2
    assert max_profit_brute([3, 3, 3, 3]) == 0
    assert max_profit_brute([1]) == 0

    print("All Best Time to Buy/Sell Stock tests passed!")
    return


@app.cell
def _(mo):
    mo.md("""
    ### Complexity Analysis — Best Time to Buy and Sell Stock

    | Approach  | Time  | Space | Why                                       |
    |-----------|-------|-------|-------------------------------------------|
    | Brute     | O(n²) | O(1)  | Every buy/sell pair checked               |
    | Optimized | O(n)  | O(1)  | Single pass tracking running minimum      |

    **Why O(n) works:** we only ever need the cheapest price seen *before* the current day.
    One variable (`min_price_so_far`) is sufficient — there's no need to look back further.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ---
    ## Problem 2 · Container With Most Water — LeetCode #11

    > Given `n` non-negative integers representing heights of vertical lines at positions `0..n-1`,
    > find two lines that together with the x-axis form a container holding the **most water**.
    >
    > `area = min(height[left], height[right]) × (right − left)`

    **Constraints**
    - `2 ≤ n ≤ 10⁵`
    - `0 ≤ height[i] ≤ 10⁴`

    **Example**
    ```
    Input:  height = [1, 8, 6, 2, 5, 4, 8, 3, 7]
    Output: 49   # lines at index 1 (h=8) and index 8 (h=7): min(8,7) × 7 = 49
    ```
    """)
    return


@app.function
def max_area_brute(height: list[int]) -> int:
    """
    Brute-force — check every pair of lines.
    Time:  O(n²)  — two nested loops
    Space: O(1)   — no extra memory
    """
    max_a = 0
    n = len(height)
    for i in range(n):
        for j in range(i + 1, n):
            area = min(height[i], height[j]) * (j - i)
            max_a = max(max_a, area)
    return max_a


@app.function
def max_area(height: list[int]) -> int:
    """
    Optimized — opposite-end two pointers.
    Time:  O(n)  — each pointer moves at most n steps total
    Space: O(1)  — two index variables

    Key insight: start with the WIDEST possible container (left=0, right=n-1).
    Always move the pointer on the SHORTER side inward.

    PROOF OF CORRECTNESS — why is this safe?
      The area is bounded (bottlenecked) by the shorter line.
      - Moving the TALLER pointer inward: width decreases AND the height limit
        cannot improve (the shorter side is still there). Area can only shrink.
        We would guarantee missing the optimal solution. ✗
      - Moving the SHORTER pointer inward: width decreases BUT we might find a
        taller line, potentially increasing area despite the narrower width.
        This is the only move that can possibly improve things. ✓
      Therefore we never skip the optimal pair.
    """
    left, right = 0, len(height) - 1
    max_a = 0

    while left < right:
        # Compute water area for the current pair of walls
        area = min(height[left], height[right]) * (right - left)
        max_a = max(max_a, area)

        # Move the shorter wall — the only candidate for improvement
        if height[left] <= height[right]:
            left += 1    # left is the bottleneck; advance to try a taller left wall
        else:
            right -= 1   # right is the bottleneck; retreat to try a taller right wall

    return max_a


@app.cell
def container_visualization():
    import matplotlib as _matplotlib
    _matplotlib.use("Agg")
    import matplotlib.pyplot as _plt

    height = [1, 8, 6, 2, 5, 4, 8, 3, 7]
    n = len(height)
    indices = list(range(n))

    left_opt, right_opt = 1, 8                                     # optimal pair
    water_level = min(height[left_opt], height[right_opt])         # 7

    _fig, _ax = _plt.subplots(figsize=(9, 5))
    _fig.patch.set_facecolor("white")
    _ax.set_facecolor("white")

    # All bars — gray by default, blue for the optimal pair
    bar_colors = ["#bdc3c7"] * n
    bar_colors[left_opt] = "#2980b9"
    bar_colors[right_opt] = "#2980b9"

    _ax.bar(indices, height, color=bar_colors, edgecolor="white",
           linewidth=0.8, width=0.8, zorder=2)

    # Shade the water volume between the two optimal walls
    water_xs = [left_opt - 0.4, left_opt - 0.4, right_opt + 0.4, right_opt + 0.4]
    water_ys = [0, water_level, water_level, 0]
    _ax.fill(water_xs, water_ys, alpha=0.25, color="#3498db",
            zorder=1, label="Water area")

    # Horizontal water-level line
    _ax.hlines(water_level, left_opt - 0.4, right_opt + 0.4,
              colors="#3498db", linewidths=2.2, linestyles="--", zorder=3,
              label=f"Water level = min(8, 7) = {water_level}")

    # Central annotation
    mid_x = (left_opt + right_opt) / 2
    _ax.text(
        mid_x, water_level / 2,
        f"width = {right_opt - left_opt}\nheight = min(8,7) = {water_level}\narea = 49",
        ha="center", va="center", fontsize=9, fontweight="bold", color="#1a5276",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85,
                  edgecolor="#3498db")
    )

    # Label the optimal bars
    _ax.text(left_opt, height[left_opt] + 0.25, "L\nh=8",
            ha="center", fontsize=8.5, color="#1a5276", fontweight="bold")
    _ax.text(right_opt, height[right_opt] + 0.25, "R\nh=7",
            ha="center", fontsize=8.5, color="#1a5276", fontweight="bold")

    _ax.set_xticks(indices)
    _ax.set_xticklabels([f"i={i}\nh={h}" for i, h in zip(indices, height)])
    _ax.set_ylabel("Height")
    _ax.set_ylim(0, 11)
    _ax.set_title(
        "LC #11 — Container With Most Water\n"
        "[1, 8, 6, 2, 5, 4, 8, 3, 7]  →  area = 49",
        fontsize=11
    )
    _ax.legend(loc="upper left", fontsize=8)
    _ax.grid(axis="y", alpha=0.3)
    _fig.tight_layout()
    return _fig


@app.cell
def _(mo):
    mo.md("""
    ### Pointer Movement Trace — `[1, 8, 6, 2, 5, 4, 8, 3, 7]`

    | Step | Left | H\\[L\\] | Right | H\\[R\\] | Width | Area   | Max    | Move                     |
    |------|------|---------|-------|---------|-------|--------|--------|--------------------------|
    | 1    | 0    | 1       | 8     | 7       | 8     | 8      | 8      | L→ (1 ≤ 7)              |
    | 2    | 1    | 8       | 8     | 7       | 7     | **49** | **49** | ←R (7 < 8)              |
    | 3    | 1    | 8       | 7     | 3       | 6     | 18     | 49     | ←R (3 < 8)              |
    | 4    | 1    | 8       | 6     | 8       | 5     | 40     | 49     | L→ (8 ≤ 8, tie → L)    |
    | 5    | 2    | 6       | 6     | 8       | 4     | 24     | 49     | L→ (6 ≤ 8)             |
    | 6    | 3    | 2       | 6     | 8       | 3     | 6      | 49     | L→ (2 ≤ 8)             |
    | 7    | 4    | 5       | 6     | 8       | 2     | 10     | 49     | L→ (5 ≤ 8)             |
    | 8    | 5    | 4       | 6     | 8       | 1     | 4      | 49     | L→ (4 ≤ 8)             |
    | —    | L = R = 6 |   |       |         |       |        | **49** | done                     |

    The optimal answer (area=49) is locked in at **step 2** (left=1, right=8).
    Every step after that narrows the window — the algorithm is exhaustive yet linear.
    """)
    return


@app.cell
def _():
    # --- Container With Most Water — test suite ---

    assert max_area([1, 8, 6, 2, 5, 4, 8, 3, 7]) == 49    # classic example
    assert max_area([1, 1]) == 1                             # minimum two-bar case
    assert max_area([4, 3, 2, 1, 4]) == 16                  # optimal uses both ends: min(4,4)*4
    assert max_area([1, 2, 1]) == 2                          # symmetric three-bar
    assert max_area([1, 8, 6, 2, 5, 4, 8, 25, 7]) == 49    # tall inner bar doesn't help: pair (1,8) still wins

    assert max_area_brute([1, 8, 6, 2, 5, 4, 8, 3, 7]) == 49
    assert max_area_brute([1, 1]) == 1
    assert max_area_brute([4, 3, 2, 1, 4]) == 16
    assert max_area_brute([1, 2, 1]) == 2
    assert max_area_brute([1, 8, 6, 2, 5, 4, 8, 25, 7]) == 49

    print("All Container With Most Water tests passed!")
    return


@app.cell
def _(mo):
    mo.md("""
    ### Complexity Analysis — Container With Most Water

    | Approach | Time  | Space | Why                                           |
    |----------|-------|-------|-----------------------------------------------|
    | Brute    | O(n²) | O(1)  | Every pair of lines checked                   |
    | Two Ptr  | O(n)  | O(1)  | Each pointer moves at most n steps total      |

    **Why O(n) works:** the two pointers start n−1 apart and each step closes the gap by exactly 1.
    After at most n−1 steps they meet — all candidates that could beat the current max are evaluated.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ---
    ## Pattern Comparison

    |                    | Buy/Sell Stock (#121)                          | Container With Most Water (#11)                    |
    |--------------------|------------------------------------------------|----------------------------------------------------|
    | **Pattern**        | Sliding window (same-direction)                | Two pointers (opposite-end, converging)            |
    | **Pointer move**   | Right always scans forward; left jumps to min  | Move the shorter side inward                       |
    | **Key insight**    | Only need the running minimum seen so far      | Safe to discard the shorter side — can't do better |
    | **State tracked**  | `min_price_so_far`, `best_profit`              | `max_area`                                         |
    | **Why O(n) works** | Each element visited exactly once              | Each pointer moves at most n steps total           |
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ---
    ## Related Problems to Practice

    ### Sliding Window (same-direction two pointers with state)
    | Problem | LC # | Note |
    |---------|------|------|
    | Longest Substring Without Repeating Characters | #3 | Classic variable-length window; shrink when duplicate enters |
    | Minimum Window Substring | #76 | Expand right until valid, then shrink left — two-phase window |
    | Max Consecutive Ones III | #1004 | Window with a "budget": at most K zeros flipped |

    ### Two Pointers — Opposite Ends ⭐ *high interview frequency*
    | Problem | LC # | Note |
    |---------|------|------|
    | **3Sum** | **#15** | ⭐ Sort + outer loop + inner two-pointer scan; very common |
    | **Trapping Rain Water** | **#42** | ⭐ Track running max from both sides; same move logic as Container |

    ### Two Pointers — Same Direction
    | Problem | LC # | Note |
    |---------|------|------|
    | Remove Duplicates from Sorted Array | #26 | Slow pointer marks next write position |
    | Move Zeroes | #283 | Slow pointer tracks next non-zero slot |

    > **Prioritize 3Sum (#15) and Trapping Rain Water (#42)** — these appear constantly in
    > FAANG screens and build directly on the patterns practiced today.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ---
    ## Interview Talking Points

    ### "Walk me through Container With Most Water" — 45-second scripted answer
    > "The brute force checks every pair in O(n²). To optimize, I notice that area is always
    > limited by the shorter of the two walls. So I start with the widest container — pointers
    > at both ends. Each step I move the shorter pointer inward, because moving the taller one
    > can only reduce width without possibly raising the height limit. Moving the shorter one
    > might find a taller wall and increase area despite the narrower width. That's the proof
    > of correctness: we never skip the optimal pair. After at most n−1 moves the pointers
    > meet — O(n) time, O(1) space."

    ---

    ### "Why not use a sliding window for Container With Most Water?"
    > "Sliding window is for contiguous subarrays where I track running state as the window
    > grows and shrinks. Here I'm not processing a contiguous range — I'm comparing two
    > specific endpoints. Opposite-end two pointers is the right model because I need to
    > systematically eliminate pairs, not maintain a window with incremental state."

    ---

    ### "What's the difference between sliding window and two pointers?"
    > "Two pointers is the broader technique — two indices traversing a structure to avoid
    > O(n²) pair enumeration. Sliding window is one specific application: same-direction two
    > pointers where you maintain a contiguous window and track running state (sum, character
    > counts, etc.). All sliding windows are two-pointer problems, but not the reverse."

    ---

    ### Follow-up for Buy/Sell Stock: "What if you can do multiple transactions?"
    > "That's LC #122. With unlimited transactions, greedy works: sum every consecutive
    > ascending pair — `sum(max(0, prices[i+1] - prices[i]) for i in range(n-1))`.
    > It's a different pattern from the sliding window here — no pointer tracking needed."

    ---

    ### Connection to real work
    > "The sliding window pattern shows up directly in my backtesting engine — scanning a
    > price series with a trailing window to compute rolling indicators (20-day moving
    > average, rolling volatility) is the same idea: maintain running state as the window
    > advances, update incrementally rather than recomputing from scratch each bar."
    """)
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
