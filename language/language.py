import numpy as np


class Node:
    def wave(self):
        raise Exception("Abstract method")

    def position(self):
        return self.wave() * self.wave()

    def distance_to(self, node):
        return node.wave() * node.wave()


class LeafNode(Node):
    def __init__(self, wave):
        self._wave = wave

    def iterate(self, energy_in):
        old_position = self.position()
        real, imag = self._wave + energy_in
        return np.sqrt(abs(self.position() - old_position))

    def wave(self):
        return self._wave


class TreeNode:
    def __init__(self, children, biases=None):
        self.children = children
        self.biases = biases if biases is not None else [1] * len(children)

    def iterate(self, energy_in):
        # Normalize biases
        total_bias = sum(self.biases)
        normalized_biases = [bias / total_bias for bias in self.biases]

        # Distribute energy among children based on biases
        total_energy_out = 0
        for child, bias in zip(self.children, normalized_biases):
            energy_out = child.iterate(energy_in * bias)
            print("child energy", child.wave())
            total_energy_out += energy_out

        # Return total energy out from all children
        return total_energy_out

    def wave_function(self):
        # Sum of all children's wave functions
        return sum(child.wave_function() for child in self.children)


class Game:
    def __init__(self, initial_energy, root_node):
        self.energy = initial_energy
        self.root_node = root_node

    def play(self, time_steps):
        for t in range(time_steps):
            # Start recursive play from the root node
            self.energy = self.root_node.iterate(self.energy)
            print("energy in system is now", self.energy)

    def is_stable(self, threshold=0.001):
        instable = 0
        while True:
            energy = self.root_node.iterate(self.energy)
            if self.energy < energy:
                instable += 1
                if instable >= 3:
                    return
            elif self.energy - energy < threshold:
                return True


# Example usage
leaf1 = LeafNode(complex(1, 0))
leaf2 = LeafNode(complex(1, 1))

# Define biases for each child

biases = [0.5, 0.5]
tree_node = TreeNode(children=[leaf1, leaf2], biases=biases)
game = Game(initial_energy=1, root_node=tree_node)
game.play(time_steps=10)

# Example of calculating distances
print("Distance from leaf1 to leaf2:", leaf1.distance_to(leaf2))
print("Distance from leaf1 to leaf2:", leaf2.distance_to(leaf1))
