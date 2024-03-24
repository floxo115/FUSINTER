from typing import List


class OrderedHeapElement:
    def __init__(self, previous_idx: int, next_idx: int, value: float, data: any = None):
        self.previous_idx = previous_idx
        self.next_idx = next_idx
        self.value = value
        self.data = data

    def __lt__(self, other):
        for i in range(len(self.value)):
            if self.value[i] < other.value[i]:
                return True
            elif self.value[i] == other.value[i]:
                continue
            else:
                return False

    def __gt__(self, other):
        for i in range(len(self.value)):
            if self.value[i] > other.value[i]:
                return True
            elif self.value[i] == other.value[i]:
                continue
            else:
                return False

    def __repr__(self):
        return f"<OrderedHeapElement [previous_idx: {self.previous_idx}, next_idx: {self.next_idx}, value: {self.value}]>"


class OrderedHeap():
    def __init__(self):
        self.array: List[OrderedHeapElement] = []

    def __len__(self):
        return len(self.array)

    def __repr__(self):
        s = f"<OrderedHeap [\n"
        for el in self.array:
            s += f"\t{el}\n"
        s += f"]>"
        return s

    @staticmethod
    def get_parent(pos: int):
        assert pos >= 0

        if pos == 0:
            return None
        if pos % 2 == 0:
            return (pos - 1) // 2
        else:
            return pos // 2

    @staticmethod
    def get_left_child(pos):
        assert pos >= 0
        return 2 * pos + 1

    def swap_elements(self, child_pos):
        # if pos_child is the root position raise a ValueError
        if child_pos == 0:
            raise ValueError("The root element can not be swapped with its parent!")

        parent_pos = OrderedHeap.get_parent(child_pos)
        parent = self.array[parent_pos]
        child = self.array[child_pos]

        # we have to check if the elements are in a neighbor
        # relationship and set the indexes according to it
        if parent.previous_idx == child_pos:
            parent.previous_idx = parent_pos
        elif parent.next_idx == child_pos:
            parent.next_idx = parent_pos

        if child.previous_idx == parent_pos:
            child.previous_idx = child_pos
        elif child.next_idx == parent_pos:
            child.next_idx = child_pos

        # we update the previous/next indices of neighbors if they are
        # not the parent/child itself (i.e. not already changed in previous step)
        if parent.previous_idx >= 0 and parent.previous_idx != parent_pos:
            self.array[parent.previous_idx].next_idx = child_pos
        if parent.next_idx >= 0 and parent.next_idx != parent_pos:
            self.array[parent.next_idx].previous_idx = child_pos

        if child.previous_idx >= 0 and child.previous_idx != child_pos:
            self.array[child.previous_idx].next_idx = parent_pos
        if child.next_idx >= 0 and child.next_idx != child_pos:
            self.array[child.next_idx].previous_idx = parent_pos

        # we swap the parent with the child inside the heap
        self.array[parent_pos], self.array[child_pos] = \
            self.array[child_pos], self.array[parent_pos]

        return parent_pos

    def insert(self, previous_idx: int, value: float, data=None):
        # create the element to be inserted into the heap
        new_element = OrderedHeapElement(previous_idx, -1, value, data)

        # the new element will be the last in the heap
        self.array.append(new_element)

        # if there are already elements the last one that was inserted has to be set to be the neighbor
        # of the new one
        if len(self) > 1:
            self.array[previous_idx].next_idx = len(self) - 1

        # the new element has to bubble upwards as long as its parent element is smaller than it
        pos = len(self) - 1
        while pos != 0:
            parent_pos = OrderedHeap.get_parent(pos)
            parent = self.array[parent_pos]
            if parent > new_element:
                break

            pos = self.swap_elements(pos)

        return pos

    def where_is_element(self, element: OrderedHeapElement):
        """
        helper to find the position of an element in the heap (could be done via saving it in the element itself...)
        """
        if element.previous_idx != -1:
            return self.array[element.previous_idx].next_idx
        elif element.next_idx != -1:
            return self.array[element.next_idx].previous_idx
        else:
            return 0

    def heapify(self, pos):
        """
        takes the position of an element in the heap and then bubbles up or down until it is in the correct position
        helps to implement delete and is useful when values of elements in the heap are updated
        """

        def _bubble_up(pos):
            # do nothing for the root element
            if pos == 0:
                return pos

            # swap position with parent elment as long as it is smaller than elment at pos
            parent_pos = OrderedHeap.get_parent(pos)
            if self.array[parent_pos] < self.array[pos]:
                self.swap_elements(pos)
                # do the same thing recursively with new position
                pos = _bubble_up(parent_pos)

            return pos

        def _bubble_down(pos):
            left_child_pos = OrderedHeap.get_left_child(pos)

            # do nothing for leaf elements
            if left_child_pos >= len(self):
                return pos

            # if there are two children we want to find the position of the larger one
            max_child_pos = left_child_pos
            right_child_pos = left_child_pos + 1
            if right_child_pos < len(self) and self.array[left_child_pos] < self.array[right_child_pos]:
                max_child_pos = right_child_pos

            # swap position if the larger child is larger than element at position
            if self.array[max_child_pos] > self.array[pos]:
                self.swap_elements(max_child_pos)
                # do the same thing recursively with new position
                pos = _bubble_down(max_child_pos)

            return pos

        pos = _bubble_up(pos)
        return _bubble_down(pos)

    def delete_element(self, pos):
        assert 0 <= pos <= len(self) - 1

        if len(self) == 0:
            return

        to_delete = self.array[pos]
        last_el = self.array[-1]

        if to_delete.previous_idx != -1:
            self.array[to_delete.previous_idx].next_idx = to_delete.next_idx
        if to_delete.next_idx != -1:
            self.array[to_delete.next_idx].previous_idx = to_delete.previous_idx

        self.array.pop()

        if to_delete is last_el:
            return

        if last_el.previous_idx != -1:
            self.array[last_el.previous_idx].next_idx = pos
        if last_el.next_idx != -1:
            self.array[last_el.next_idx].previous_idx = pos

        self.array[pos] = last_el
        self.heapify(pos)
