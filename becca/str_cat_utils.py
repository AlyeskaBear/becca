"""
Utility functions for working with lists of strings.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


def variance(names):
    """
    Determine a variance-like quantity for a set of strings.

    @param names: dict of <string, int>
        A set strings to find variance for, and their associated counts.
    @return: float
        A variance-like measure of how distributed the items are.
    """
    # In case there is only one name or no names.
    if len(names.keys()) < 2:
        return 0.

    total_distance = 0.
    sum_counts = 0.
    for name_a, count_a in names.items():
        sum_counts += count_a
        for name_b, count_b in names.items():
            if name_a != name_b:
                # TODO: Create a more sophisticated measure of
                # the distance between two strings.
                # In it's current form it is
                #     0 if the strings are identical
                #     1 if the strings aren't identical
                distance_a_b = 1.
                total_distance += distance_a_b * count_a * count_b
    return total_distance / sum_counts
