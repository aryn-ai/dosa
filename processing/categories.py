from abc import abstractmethod
from enum import Enum


class Category(Enum):
    Caption = 1
    Footnote = 2
    Formula = 3
    ListItem = 4
    PageFooter = 5
    PageHeader = 6
    Picture = 7
    SectionHeader = 8
    Table = 9
    Text = 10
    Title = 11
    Section = 101


class Node:
    def __init__(self, node_id, category, metadata):
        self._node_id = node_id
        self._category = category
        self._metadata = metadata

    @property
    def node_id(self):
        return self._node_id

    @abstractmethod
    def category(self): ...

    @property
    def metadata(self):
        return self._metadata


class Leaf(Node):
    pass


class Internal(Node):
    def children(self) -> list[Node]:
        pass


class Caption(Leaf):
    def category(self):
        return Category.Caption


class Footnote(Leaf):
    def category(self):
        return Category.Footnote


class Formula(Leaf):
    def category(self):
        return Category.Formula


class ListItem(Leaf):
    def category(self):
        return Category.ListItem


class PageFooter(Leaf):
    def category(self):
        return Category.PageFooter


class PageHeader(Leaf):
    def category(self):
        return Category.PageHeader


class Picture(Leaf):
    def category(self):
        return Category.Picture


class SectionHeader(Leaf):
    def category(self):
        return Category.SectionHeader


class Table(Leaf):
    def category(self):
        return Category.Table


class Text(Leaf):
    def category(self):
        return Category.Text


class Title(SectionHeader):
    def category(self):
        return Category.Title


class Section(Internal):
    def category(self):
        return Category.Section
