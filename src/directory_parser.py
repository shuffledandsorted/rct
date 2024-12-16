import os
from custom_node import CustomNode
from visitors import DirectoryVisitor


def parse_directory_structure(root_dir, parent_node=None):
    if parent_node is None:
        parent_node = CustomNode(root_dir)

    # Use DirectoryVisitor to process the entire directory structure
    directory_visitor = DirectoryVisitor(parent_node)
    for dirpath, dirnames, filenames in os.walk(root_dir, topdown=True):
        print(dirpath)
        directory_visitor.visit(dirpath, dirnames, filenames)

    return parent_node
