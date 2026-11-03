"""Supplement link labels and immutable sampling-stream indices."""

CASE_ORDER = ("small_rx", "strong_small_rx", "long_rx20")
CASE_LABELS = {case: f"L{index + 1}" for index, case in enumerate(CASE_ORDER)}

# These indices determine archived random streams, independently of display order.
SAMPLING_CASE_INDEX = {"small_rx": 0, "long_rx20": 1, "strong_small_rx": 2}
