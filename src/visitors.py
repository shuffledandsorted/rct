import ast
import re
from typing import List, Union
from custom_node import CustomNode  # Import your custom node class


class BaseVisitor:
    """Base class for visitors that traverse and process nodes.

    Subclasses should implement the visit method to handle specific node types.
    """

    def visit(self, node: Union[ast.AST, str, tuple]) -> None:
        raise NotImplementedError("Subclasses should implement this method.")


class ASTVisitor(BaseVisitor):
    """Visitor for traversing and processing AST nodes.

    This class creates nodes in the tree structure using anytree. Each node is
    automatically added to the tree when created, establishing parent-child
    relationships based on the AST structure.
    """

    def __init__(self, parent_node: CustomNode) -> None:
        self.parent_node = parent_node

    def visit(self, node: ast.AST) -> None:
        method_name = f"visit_{node.__class__.__name__}"
        visitor = getattr(self, method_name, self.generic_visit)
        visitor(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        relevance_score = 1.0  # Initial score for functions
        relevance_score *= self.parent_node.relevance_score  # Propagate score

        CustomNode(
            node.name,
            parent=self.parent_node,
            node_type="Function",
            relevance_score=relevance_score,
        )
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        relevance_score = 1.0  # Initial score for classes
        relevance_score *= self.parent_node.relevance_score  # Propagate score

        CustomNode(
            node.name,
            parent=self.parent_node,
            node_type="Class",
            relevance_score=relevance_score,
        )
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            if isinstance(target, ast.Name):
                CustomNode(
                    target.id,
                    parent=self.parent_node,
                    node_type="Variable",
                    relevance_score=0.5,
                )
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            CustomNode(
                alias.name,
                parent=self.parent_node,
                node_type="Import",
                relevance_score=0.5,
            )

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for alias in node.names:
            CustomNode(
                f"{node.module}.{alias.name}",
                parent=self.parent_node,
                node_type="ImportFrom",
                relevance_score=0.5,
            )

    def generic_visit(self, node: ast.AST) -> None:
        for child in ast.iter_child_nodes(node):
            self.visit(child)


class TextFileVisitor(BaseVisitor):
    """Visitor for processing text file content.

    This class creates nodes for headers found in text files, adding them to
    the tree structure.
    """

    def __init__(self, parent_node: CustomNode) -> None:
        self.parent_node = parent_node

    def visit(self, content: str) -> None:
        headers = self.parse_text_file(content)
        for header in headers:
            relevance_score = 0.5  # Initial score for headers
            relevance_score *= self.parent_node.relevance_score  # Propagate score

            CustomNode(
                header,
                parent=self.parent_node,
                node_type="Header",
                relevance_score=relevance_score,
            )

    def parse_text_file(self, content: str) -> List[str]:
        headers = []
        for line in content.splitlines():
            if line.startswith("#"):
                headers.append(line.strip())
            elif re.match(r"^[A-Za-z0-9\s]+[:\-]", line):
                headers.append(line.strip())
        return headers


class DirectoryVisitor(BaseVisitor):
    """Visitor for processing directory structures.

    This class creates nodes for directories and files, adding them to the
    tree structure.
    """

    def __init__(self, parent_node: CustomNode) -> None:
        self.parent_node = parent_node

    def visit(self, dirpath: str, dirnames: List[str], filenames: List[str]) -> None:
        # Process directories
        for dirname in dirnames:
            if dirname.startswith(".") and dirname != ".":
                relevance_score = 0.5
            else:
                relevance_score = 2.0

            relevance_score *= self.parent_node.relevance_score
            if relevance_score < 1:
                continue

            CustomNode(
                dirname,
                parent=self.parent_node,
                node_type="Directory",
                relevance_score=relevance_score,
            )

        # Process files
        for filename in filenames:
            if filename.endswith(".py"):
                node_type = "Python File"
                relevance_score = 1.5
            elif filename.endswith(".md"):
                node_type = "Markdown File"
                relevance_score = 1.0
            elif filename.endswith(".txt"):
                node_type = "Text File"
                relevance_score = 1.0
            elif filename.endswith(".html"):
                node_type = "HTML File"
                relevance_score = 1.0
            else:
                node_type = "Other File"
                relevance_score = 0.5

            relevance_score *= self.parent_node.relevance_score
            if relevance_score < 0.5:
                continue

            CustomNode(
                filename,
                parent=self.parent_node,
                node_type=node_type,
                relevance_score=relevance_score,
            )
