# Projects/Risk_Assessment_Models/backend/core/algorithms/sorting.py

import math


class SortingAlgorithms:
    """
    A container for various sorting algorithms.
    """

    @staticmethod
    def merge_sort(arr: list, key=lambda x: x) -> list[float]:
        """
        Merge Sort
        Sorts a list of floats using the merge sort algorithm.
        """
        if len(arr) <= 1:
            return arr

        mid = len(arr) // 2
        left_half = SortingAlgorithms.merge_sort(arr[:mid], key)
        right_half = SortingAlgorithms.merge_sort(arr[mid:], key)

        sorted_arr = []
        i, j = 0, 0
        while i < len(left_half) and j < len(right_half):
            if key(left_half[i]) < key(right_half[j]):
                sorted_arr.append(left_half[i])
                i += 1
            else:
                sorted_arr.append(right_half[j])
                j += 1
        sorted_arr.extend(left_half[i:])
        sorted_arr.extend(right_half[j:])
        return sorted_arr

    @staticmethod
    def heap_sort(arr: list, key=lambda x: x) -> list:
        """
        Heap Sort
        """
        arr_copy = list(arr)
        n = len(arr_copy)

        def _heapify(sub_arr, size, root_idx):
            largest = root_idx
            left = 2 * root_idx + 1
            right = 2 * root_idx + 2

            if left < size and key(sub_arr[left]) > key(sub_arr[largest]):
                largest = left
            if right < size and key(sub_arr[right]) > key(sub_arr[largest]):
                largest = right

            if largest != root_idx:
                sub_arr[root_idx], sub_arr[largest] = sub_arr[largest], sub_arr[root_idx]
                _heapify(sub_arr, size, largest)

        for i in range(n // 2 - 1, -1, -1):
            _heapify(arr_copy, n, i)

        for i in range(n - 1, 0, -1):
            arr_copy[i], arr_copy[0] = (
                arr_copy[0],
                arr_copy[i],
            )  # Move the maximum element to the end
            _heapify(arr_copy, i, 0)

        return arr_copy

    @staticmethod
    def msd_radix_sort(arr: list) -> list:
        """
        MSD Radix Sort (Most Significant Digit)
        """
        if not arr:
            return []

        max_val = max(arr)
        max_digits = int(math.log10(max_val)) + 1 if max_val > 0 else 1

        def _msd_sort_recursive(sub_arr, digit_idx):
            if not sub_arr or len(sub_arr) <= 1 or digit_idx >= max_digits:
                return sub_arr

            buckets = [[] for _ in range(10)]

            divisor = 10 ** (max_digits - 1 - digit_idx)
            for num in sub_arr:
                digit = (num // divisor) % 10
                buckets[digit].append(num)

            sorted_arr = []
            for bucket in buckets:
                sorted_arr.extend(_msd_sort_recursive(bucket, digit_idx + 1))

            return sorted_arr

        return _msd_sort_recursive(list(arr), 0)
