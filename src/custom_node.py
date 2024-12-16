from anytree import NodeMixin


class CustomNode(NodeMixin):
    def __init__(self, name, parent=None, node_type=None, relevance_score=1.0):
        self.name = name
        self.parent = parent
        self.node_type = node_type
        self.relevance_score = relevance_score
