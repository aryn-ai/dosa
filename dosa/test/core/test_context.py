from dosa.core.context import build_context
from dosa.core.node import Node, Tree


def test_single_node_context():
    root = Node(0, category_id=0)
    node1 = Node(1, page_id=0, order=1, category_id=10, content="Title A")
    node1.parent = root
    root.children.append(node1)

    tree = Tree({0: root, 1: node1})
    pages, count = build_context(tree, window=5)

    assert count == 1
    assert len(pages) == 1
    assert pages[0]["page_id"] == 0
    assert pages[0]["objects"][0]["id"] == 1
    assert pages[0]["objects"][0]["parent_id"] == 0


def test_non_hierarchy_nodes_ignored():
    root = Node(0, category_id=0)
    node1 = Node(
        1, page_id=0, order=1, category_id=5, content="Footer"
    )  # non-hierarchy
    node1.parent = root
    root.children.append(node1)

    tree = Tree({0: root, 1: node1})
    pages, count = build_context(tree, window=5)

    assert count == 0
    assert pages == []


def test_chain_hierarchy_nodes():
    root = Node(0, category_id=0)
    n1 = Node(1, page_id=0, order=1, category_id=10, content="Title")
    n2 = Node(2, page_id=0, order=2, category_id=12, content="Subtitle")
    n3 = Node(3, page_id=0, order=3, category_id=18, content="Section")

    # Set parents
    n1.parent = root
    n2.parent = n1
    n3.parent = n2
    root.children.append(n1)
    n1.children.append(n2)
    n2.children.append(n3)

    tree = Tree({0: root, 1: n1, 2: n2, 3: n3})
    pages, count = build_context(tree, window=3)

    assert count == 3
    ids = [obj["id"] for page in pages for obj in page["objects"]]
    assert ids == [1, 2, 3]


def test_window_limit():
    root = Node(0, category_id=0)
    nodes = {}
    parent = root
    for i in range(1, 10):
        node = Node(i, page_id=0, order=i, category_id=10, content=f"Title {i}")
        node.parent = parent
        parent.children.append(node)
        parent = node
        nodes[i] = node

    nodes[0] = root
    tree = Tree(nodes)

    pages, count = build_context(tree, window=5)
    assert count == 9
    ids = [obj["id"] for page in pages for obj in page["objects"]]
    assert len(ids) == 9
    assert set(ids).issubset(set(range(1, 10)))


def test_mixed_nodes():
    root = Node(0, category_id=0)

    n1 = Node(
        1, page_id=0, order=1, category_id=5, content="Footer"
    )  # non-hierarchy
    n2 = Node(
        2, page_id=0, order=2, category_id=10, content="Title"
    )  # hierarchy
    n3 = Node(
        3, page_id=0, order=3, category_id=7, content="Page Num"
    )  # non-hierarchy
    n4 = Node(
        4, page_id=0, order=4, category_id=12, content="Subtitle"
    )  # hierarchy

    n1.parent = root
    n2.parent = root
    n3.parent = n2
    n4.parent = n2

    root.children.extend([n1, n2])
    n2.children.extend([n3, n4])

    tree = Tree({0: root, 1: n1, 2: n2, 3: n3, 4: n4})

    pages, count = build_context(tree, window=3)

    assert count == 2  # Only n2 and n4 are included
    object_ids = [obj["id"] for page in pages for obj in page["objects"]]
    assert set(object_ids) == {2, 4}
