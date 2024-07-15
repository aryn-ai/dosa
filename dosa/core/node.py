from PIL import Image
from apted import APTED, Config


class Node:
    def __init__(
        self,
        name: int,
        page_id: int = -1,
        order=-1,
        bbox: list[int] = None,
        category_id: int = None,
        content: str = None,
    ):
        """
        :param name: node id, 0 is always the dummy root
        :param page_id: page id
        :param order: reading order of page object, for all metadata objects
            like header, footer, it's 0, for all other objects, the starting
            base number is 1, but absolute number is not required, a relative
            number is good enough, i.e. it does not need to be contiguous
        :param bbox: bbox in terms of dimension 800 x 800
        :param category_id: category number
        :param content: semantic content of the page object
        """
        self.name = name
        self.page_id = page_id
        self.order = order
        self.bbox = bbox
        self.category = category_id
        self.content = content
        self.parent = None
        self.children = []

    def __len__(self):
        length = 1
        for child in self.children:
            length += len(child)
        return length

    def non_hierarchy(self):
        # footer, header, page-number, sidebar, footnote
        return self.category in [5, 6, 7, 9, 11]
        # return self.category in [5, 6, 7, 8, 9, 11]

    def hierarchy(self):
        # hierarchy labels: section-title, sub-title, title, toc-title
        return self.category in [10, 12, 18, 19]

    def potential_hierarchy(self):
        # potential hierarchy labels: figure, figure-caption, figure-title,
        # table, toc, text and table-title, they appear as pair wise parent
        # child relation
        return self.category in [0, 1, 2, 3, 4, 8, 13, 14, 15, 16, 17]

    @staticmethod
    def tree_edit_distance(predicted: "Node", gt: "Node") -> float:
        distance = APTED(predicted, gt, Config()).compute_edit_distance()
        teds = 1.0 - (float(distance) / max([len(predicted), len(gt)]))
        return teds


class Tree:
    def __init__(self, nodes: dict[int, Node], images: list[Image] = None):
        self.nodes = nodes
        self.root = nodes[0]
        self.images = images

    def cycled(self):
        visited = []

        def dfs(node):
            if node.name in visited:
                return True
            visited.append(node.name)
            for child in node.children:
                if dfs(child):
                    return True
            return False

        dfs(self.root)
