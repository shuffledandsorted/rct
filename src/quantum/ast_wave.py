"""Represent Abstract Syntax Trees as quantum wave functions.

The key idea is that code can exist in a superposition of possible AST structures,
where phase relationships encode structural dependencies.
"""

import ast
import numpy as np
from typing import List, Optional, Dict, Any, Type, Union

# Define valid AST node types
NodeType = Union[
    Type[ast.Name],
    Type[ast.Call],
    Type[ast.Attribute],
    Type[ast.BinOp],
    Type[ast.Compare],
    Type[ast.If],
    Type[ast.For],
    Type[ast.While],
    Type[ast.FunctionDef],
]


class ASTWaveFunction:
    def __init__(self, dims: tuple[int, int]):
        """Initialize a wave function for AST representation.

        Args:
            dims: The dimensions of the wave function grid
                 (can be thought of as max depth x max branching factor)
        """
        self.dims = dims
        self.amplitude = np.zeros(dims, dtype=np.complex128)
        self.node_types: Dict[NodeType, float] = {
            # Core node types with their base phases
            ast.Name: 0,
            ast.Call: np.pi / 4,
            ast.Attribute: np.pi / 2,
            ast.BinOp: 3 * np.pi / 4,
            ast.Compare: np.pi,
            ast.If: 5 * np.pi / 4,
            ast.For: 3 * np.pi / 2,
            ast.While: 7 * np.pi / 4,
            ast.FunctionDef: 2 * np.pi,
        }

    def encode_node(self, node: ast.AST, depth: int = 0, pos: int = 0) -> None:
        """Encode an AST node into the wave function."""
        if depth >= self.dims[0] or pos >= self.dims[1]:
            return

        # Base amplitude for this node type
        base_amp = 1.0 / (depth + 1)  # Amplitude decreases with depth

        # Phase based on node type
        node_type = type(node)
        if node_type in self.node_types:
            phase = self.node_types[node_type]
        else:
            phase = 0.0

        # Encode this node
        self.amplitude[depth, pos] = base_amp * np.exp(1j * phase)

        # Recursively encode children with phase relationships
        for i, child in enumerate(ast.iter_child_nodes(node)):
            child_pos = pos + i + 1
            if child_pos < self.dims[1]:
                # Add phase relationship between parent and child
                self.encode_node(child, depth + 1, child_pos)
                # Entangle parent with child through phase
                parent_phase = np.angle(self.amplitude[depth, pos])
                child_phase = np.angle(self.amplitude[depth + 1, child_pos])
                entangle_phase = (parent_phase + child_phase) / 2
                self.amplitude[depth + 1, child_pos] *= np.exp(1j * entangle_phase)

    def decode_ast(self) -> Optional[ast.AST]:
        """Collapse the wave function into a concrete AST."""
        # Start with the root (highest amplitude in depth 0)
        root_pos = int(np.argmax(np.abs(self.amplitude[0])))
        root_phase = np.angle(self.amplitude[0, root_pos])

        # Find the closest node type based on phase
        root_type = min(self.node_types.items(), key=lambda x: abs(x[1] - root_phase))[
            0
        ]

        # Recursively reconstruct the tree
        return self._decode_node(0, root_pos, root_type)

    def _decode_node(
        self, depth: int, pos: int, node_type: NodeType
    ) -> Optional[ast.AST]:
        """Recursively decode a node and its children based on phase relationships."""
        if depth >= self.dims[0] or pos >= self.dims[1]:
            return None

        # Create node of appropriate type with dummy values
        if node_type == ast.Name:
            node = ast.Name(id="var", ctx=ast.Load())
        elif node_type == ast.Call:
            node = ast.Call(
                func=ast.Name(id="func", ctx=ast.Load()), args=[], keywords=[]
            )
        elif node_type == ast.BinOp:
            node = ast.BinOp(
                left=ast.Name(id="x", ctx=ast.Load()),
                op=ast.Add(),
                right=ast.Name(id="y", ctx=ast.Load()),
            )
        elif node_type == ast.If:
            node = ast.If(test=ast.Constant(value=True), body=[], orelse=[])
        else:
            return None

        # Find children based on phase relationships
        children = []
        for i in range(self.dims[1]):
            child_pos = pos + i + 1
            if child_pos >= self.dims[1]:
                break

            child_amp = self.amplitude[depth + 1, child_pos]
            if np.abs(child_amp) > 0.1:  # Amplitude threshold
                child_phase = np.angle(child_amp)
                child_type = min(
                    self.node_types.items(), key=lambda x: abs(x[1] - child_phase)
                )[0]
                child_node = self._decode_node(depth + 1, child_pos, child_type)
                if child_node:
                    children.append(child_node)

        # Attach children based on node type
        if isinstance(node, ast.If):
            node.body = children
        elif isinstance(node, ast.Call):
            node.args = children

        return node

    def superpose(self, other: "ASTWaveFunction", alpha: float = 0.5) -> None:
        """Create a superposition of two AST wave functions."""
        if self.dims != other.dims:
            raise ValueError("Wave functions must have same dimensions")

        # Quantum superposition with phase preservation
        self.amplitude = alpha * self.amplitude + (1 - alpha) * other.amplitude
        # Normalize
        self.amplitude /= np.linalg.norm(self.amplitude)

    def measure_coherence(self) -> float:
        """Measure the structural coherence of the AST wave function."""
        # Phase coherence between adjacent nodes indicates structural validity
        coherence = 0.0
        for i in range(self.dims[0] - 1):
            for j in range(self.dims[1] - 1):
                if np.abs(self.amplitude[i, j]) > 0.1:
                    phase_diff = np.angle(self.amplitude[i + 1, j]) - np.angle(
                        self.amplitude[i, j]
                    )
                    coherence += np.cos(phase_diff)
        return coherence / (self.dims[0] * self.dims[1])
