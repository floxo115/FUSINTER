import numpy as np
import pytest

from .splitvaluecomputer import MaxHeap


class TestMaxHeap:

    @pytest.mark.parametrize("table, split_values, split_points", [
        (
                np.arange(0, 30).reshape(3, 10),
                np.array([5, 4, 6, 3, 7, 2, 8, 1, 0, 9]),
                np.arange(0, 10),
        ),
    ])
    def test_creation_and_insert(self, table, split_points, split_values):
        heap = MaxHeap()
        assert len(heap) == 0

        pos = heap.insert(table[:, 0], -1, split_points[0], split_values[0])
        assert pos == 0
        assert len(heap) == 1

        pos = heap.insert(table[:, 1], pos, split_points[1], split_values[1])
        assert pos == 1
        assert heap.array[0].next_idx == 1
        assert heap.array[pos].prev_idx == 0
        assert len(heap) == 2

        pos = heap.insert(table[:, 2], pos, split_points[2], split_values[2])
        assert pos == 0

        assert heap.array[0].split_value == 6
        assert heap.array[0].prev_idx == 1
        assert heap.array[0].next_idx == -1

        assert heap.array[1].split_value == 4
        assert heap.array[1].prev_idx == 2
        assert heap.array[1].next_idx == 0

        assert heap.array[2].split_value == 5
        assert heap.array[2].prev_idx == -1
        assert heap.array[2].next_idx == 1

        assert len(heap) == 3

        pos = heap.insert(table[:, 3], pos, split_points[3], split_values[3])
        assert pos == 3

        assert heap.array[0].split_value == 6
        assert heap.array[0].prev_idx == 1
        assert heap.array[0].next_idx == 3

        assert heap.array[1].split_value == 4
        assert heap.array[1].prev_idx == 2
        assert heap.array[1].next_idx == 0

        assert heap.array[2].split_value == 5
        assert heap.array[2].prev_idx == -1
        assert heap.array[2].next_idx == 1

        assert heap.array[3].split_value == 3
        assert heap.array[3].prev_idx == 0
        assert heap.array[3].next_idx == -1

        assert len(heap) == 4

        pos = heap.insert(table[:, 4], pos, split_points[4], split_values[4])
        assert pos == 0

        assert heap.array[0].split_value == 7
        assert heap.array[0].prev_idx == 3
        assert heap.array[0].next_idx == -1

        assert heap.array[1].split_value == 6
        assert heap.array[1].prev_idx == 4
        assert heap.array[1].next_idx == 3

        assert heap.array[2].split_value == 5
        assert heap.array[2].prev_idx == -1
        assert heap.array[2].next_idx == 4

        assert heap.array[3].split_value == 3
        assert heap.array[3].prev_idx == 1
        assert heap.array[3].next_idx == -0

        assert heap.array[4].split_value == 4
        assert heap.array[4].prev_idx == 2
        assert heap.array[4].next_idx == 1

        assert len(heap) == 5

        pos = heap.insert(table[:, 5], pos, split_points[5], split_values[5])
        assert pos == 5

        assert heap.array[0].split_value == 7
        assert heap.array[0].prev_idx == 3
        assert heap.array[0].next_idx == 5

        assert heap.array[1].split_value == 6
        assert heap.array[1].prev_idx == 4
        assert heap.array[1].next_idx == 3

        assert heap.array[2].split_value == 5
        assert heap.array[2].prev_idx == -1
        assert heap.array[2].next_idx == 4

        assert heap.array[3].split_value == 3
        assert heap.array[3].prev_idx == 1
        assert heap.array[3].next_idx == 0

        assert heap.array[4].split_value == 4
        assert heap.array[4].prev_idx == 2
        assert heap.array[4].next_idx == 1

        assert heap.array[5].split_value == 2
        assert heap.array[5].prev_idx == 0
        assert heap.array[5].next_idx == -1

        assert len(heap) == 6

        pos = heap.insert(table[:, 6], pos, split_points[6], split_values[6])
        assert pos == 0

        assert heap.array[0].split_value == 8
        assert heap.array[0].prev_idx == 5
        assert heap.array[0].next_idx == -1

        assert heap.array[1].split_value == 6
        assert heap.array[1].prev_idx == 4
        assert heap.array[1].next_idx == 3

        assert heap.array[2].split_value == 7
        assert heap.array[2].prev_idx == 3
        assert heap.array[2].next_idx == 5

        assert heap.array[3].split_value == 3
        assert heap.array[3].prev_idx == 1
        assert heap.array[3].next_idx == 2

        assert heap.array[4].split_value == 4
        assert heap.array[4].prev_idx == 6
        assert heap.array[4].next_idx == 1

        assert heap.array[5].split_value == 2
        assert heap.array[5].prev_idx == 2
        assert heap.array[5].next_idx == 0

        assert heap.array[6].split_value == 5
        assert heap.array[6].prev_idx == -1
        assert heap.array[6].next_idx == 4

        assert len(heap) == 7

        pos = heap.insert(table[:, 7], pos, split_points[7], split_values[7])
        assert pos == 7

        assert heap.array[0].split_value == 8
        assert heap.array[0].prev_idx == 5
        assert heap.array[0].next_idx == 7

        assert heap.array[1].split_value == 6
        assert heap.array[1].prev_idx == 4
        assert heap.array[1].next_idx == 3

        assert heap.array[2].split_value == 7
        assert heap.array[2].prev_idx == 3
        assert heap.array[2].next_idx == 5

        assert heap.array[3].split_value == 3
        assert heap.array[3].prev_idx == 1
        assert heap.array[3].next_idx == 2

        assert heap.array[4].split_value == 4
        assert heap.array[4].prev_idx == 6
        assert heap.array[4].next_idx == 1

        assert heap.array[5].split_value == 2
        assert heap.array[5].prev_idx == 2
        assert heap.array[5].next_idx == 0

        assert heap.array[6].split_value == 5
        assert heap.array[6].prev_idx == -1
        assert heap.array[6].next_idx == 4

        assert heap.array[7].split_value == 1
        assert heap.array[7].prev_idx == 0
        assert heap.array[7].next_idx == -1

        assert len(heap) == 8

        pos = heap.insert(table[:, 8], pos, split_points[8], split_values[8])
        assert pos == 8

        assert heap.array[0].split_value == 8
        assert heap.array[0].prev_idx == 5
        assert heap.array[0].next_idx == 7

        assert heap.array[1].split_value == 6
        assert heap.array[1].prev_idx == 4
        assert heap.array[1].next_idx == 3

        assert heap.array[2].split_value == 7
        assert heap.array[2].prev_idx == 3
        assert heap.array[2].next_idx == 5

        assert heap.array[3].split_value == 3
        assert heap.array[3].prev_idx == 1
        assert heap.array[3].next_idx == 2

        assert heap.array[4].split_value == 4
        assert heap.array[4].prev_idx == 6
        assert heap.array[4].next_idx == 1

        assert heap.array[5].split_value == 2
        assert heap.array[5].prev_idx == 2
        assert heap.array[5].next_idx == 0

        assert heap.array[6].split_value == 5
        assert heap.array[6].prev_idx == -1
        assert heap.array[6].next_idx == 4

        assert heap.array[7].split_value == 1
        assert heap.array[7].prev_idx == 0
        assert heap.array[7].next_idx == 8

        assert heap.array[8].split_value == 0
        assert heap.array[8].prev_idx == 7
        assert heap.array[8].next_idx == -1

        assert len(heap) == 9

        pos = heap.insert(table[:, 9], pos, split_points[9], split_values[9])
        assert pos == 0

        assert heap.array[0].split_value == 9
        assert heap.array[0].prev_idx == 8
        assert heap.array[0].next_idx == -1

        assert heap.array[1].split_value == 8
        assert heap.array[1].prev_idx == 5
        assert heap.array[1].next_idx == 7

        assert heap.array[2].split_value == 7
        assert heap.array[2].prev_idx == 3
        assert heap.array[2].next_idx == 5

        assert heap.array[3].split_value == 3
        assert heap.array[3].prev_idx == 4
        assert heap.array[3].next_idx == 2

        assert heap.array[4].split_value == 6
        assert heap.array[4].prev_idx == 9
        assert heap.array[4].next_idx == 3

        assert heap.array[5].split_value == 2
        assert heap.array[5].prev_idx == 2
        assert heap.array[5].next_idx == 1

        assert heap.array[6].split_value == 5
        assert heap.array[6].prev_idx == -1
        assert heap.array[6].next_idx == 9

        assert heap.array[7].split_value == 1
        assert heap.array[7].prev_idx == 1
        assert heap.array[7].next_idx == 8

        assert heap.array[8].split_value == 0
        assert heap.array[8].prev_idx == 7
        assert heap.array[8].next_idx == -0

        assert heap.array[9].split_value == 4
        assert heap.array[9].prev_idx == 6
        assert heap.array[9].next_idx == 4

        assert len(heap) == 10

    #########################################################################
    def test_insert_with_swapping_neighbors(self):
        heap = MaxHeap()
        pos = heap.insert(np.array([0, 0, 0]), -1, 0, 0)
        pos = heap.insert(np.array([0, 0, 0]), pos, 1, 1)

        assert heap.array[0].split_value == 1
        assert heap.array[0].prev_idx == 1
        assert heap.array[0].next_idx == -1

        assert heap.array[1].split_value == 0
        assert heap.array[1].prev_idx == -1
        assert heap.array[1].next_idx == -0

        pos = heap.insert(np.array([0, 0, 0]), pos, 2, 2)

        assert heap.array[0].split_value == 2
        assert heap.array[0].prev_idx == 2
        assert heap.array[0].next_idx == -1

        assert heap.array[1].split_value == 0
        assert heap.array[1].prev_idx == -1
        assert heap.array[1].next_idx == 0

        assert heap.array[2].split_value == 1
        assert heap.array[2].prev_idx == 1
        assert heap.array[2].next_idx == 0

    @pytest.mark.parametrize("table, split_values, split_points", [
        (
                np.arange(0, 30).reshape(3, 10),
                np.array([5, 4, 6, 3, 7, 2, 8, 1, 0, 9]),
                np.arange(0, 10),
        ),
    ])
    def test_heapify(self, table, split_values, split_points):
        heap = MaxHeap()

        pos = heap.insert(table[:, 0], -1, split_points[0], split_values[0])
        for i in range(1, len(split_values)):
            pos = heap.insert(table[:, i], pos, split_points[i], split_values[i])

        heap.array[0].split_value = -1
        heap.heapify(0)

        assert heap.array[0].split_value == 8
        assert heap.array[1].split_value == 6
        assert heap.array[4].split_value == 4
        assert heap.array[9].split_value == -1

        heap = MaxHeap()

        pos = heap.insert(table[:, 0], -1, split_points[0], split_values[0])
        for i in range(1, len(split_values)):
            pos = heap.insert(table[:, i], pos, split_points[i], split_values[i])

        heap.array[1].split_value = 5
        heap.heapify(1)

        assert heap.array[1].split_value == 6
        assert heap.array[4].split_value == 5
        assert heap.array[9].split_value == 4

    @pytest.mark.parametrize("table, split_values, split_points", [
        (
                np.arange(0, 30).reshape(3, 10),
                np.array([5, 4, 6, 3, 7, 2, 8, 1, 0, 9]),
                np.arange(0, 10),
        ),
    ])
    def test_delete(self, table, split_values, split_points):
        heap = MaxHeap()

        pos = heap.insert(table[:, 0], -1, split_points[0], split_values[0])
        for i in range(1, len(split_values)):
            pos = heap.insert(table[:, i], pos, split_points[i], split_values[i])

        heap.delete(1)

        assert heap.array[1].split_value == 6
        assert heap.array[4].split_value == 4

        heap = MaxHeap()

        pos = heap.insert(table[:, 0], -1, split_points[0], split_values[0])
        for i in range(1, len(split_values)):
            pos = heap.insert(table[:, i], pos, split_points[i], split_values[i])

        heap.delete(0)

        assert heap.array[0].split_value == 8
        assert heap.array[1].split_value == 6
        assert heap.array[4].split_value == 4

        heap = MaxHeap()

        pos = heap.insert(table[:, 0], -1, split_points[0], split_values[0])
        for i in range(1, len(split_values)):
            pos = heap.insert(table[:, i], pos, split_points[i], split_values[i])

        heap.delete(9)

        assert heap.array[6].next_idx == -1

        heap = MaxHeap()

        pos = heap.insert(table[:, 0], -1, split_points[0], split_values[0])

        heap.delete(0)

        assert len(heap.array) == 0
