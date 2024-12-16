import readline
from anytree import RenderTree
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from custom_node import CustomNode
from directory_parser import parse_directory_structure


class ConceptManager:

    def __init__(
        self, reference_concept: str, root_directory: str, top_k: int = 5
    ) -> None:
        self.reference_concept = reference_concept
        self.top_k = top_k
        self.root_node = CustomNode(root_directory)
        self.concepts = set()
        self.similarity_scores = {}
        self.top_k_concepts = []

        # Seed the initial concepts by parsing the directory structure
        self.seed_concepts()
        # Normalize scores after seeding
        self.propagate_relevance(self.root_node)

    def print_tree(self):
        for pre, fill, node in RenderTree(self.root_node):
            print(
                f"{pre}{
                    node.name} (Type: {
                    node.node_type}, Relevance: {
                    node.relevance_score:.2f})"
            )

    def seed_concepts(self) -> None:
        # Parse the directory structure and initialize the concept tree
        parse_directory_structure(
            self.root_node.name, parent_node=self.root_node)
        self.update_concepts(self.root_node)

    def update_concepts(self, node: CustomNode) -> None:
        # Recursively gather concepts from the current node and its children
        self.concepts.add(node.name)
        for child in node.children:
            self.update_concepts(child)

    def calculate_similarity(self) -> None:
        vectorizer = TfidfVectorizer()
        if not self.concepts:
            return
        concepts_matrix = vectorizer.fit_transform(list(self.concepts))
        reference_matrix = vectorizer.transform([self.reference_concept])
        scores = cosine_similarity(reference_matrix, concepts_matrix).flatten()

        # Assign scores to nodes
        for concept, score in zip(self.concepts, scores):
            self.similarity_scores[concept] = score

    def propagate_relevance(self, node: CustomNode) -> None:
        # Calculate total relevance for the subtree
        total_relevance = sum(child.relevance_score for child in node.children)
        if total_relevance > 0:
            for child in node.children:
                # Normalize child scores
                child.relevance_score /= total_relevance

        # Propagate relevance up to the parent
        if node.parent:
            node.parent.relevance_score += node.relevance_score * \
                (1 - total_relevance)

        # Recursively propagate relevance for each child
        for child in node.children:
            self.propagate_relevance(child)

    def update_top_k_concepts(self) -> None:
        # Update the top-K concepts based on the current similarity scores
        sorted_concepts = sorted(
            self.similarity_scores.items(), key=lambda x: x[1], reverse=True
        )
        self.top_k_concepts = sorted_concepts[: self.top_k]

    def explore_node(self, node: CustomNode) -> None:
        # Update concepts and calculate similarity for the current node
        self.update_concepts(node)
        self.calculate_similarity()
        self.update_top_k_concepts()

    def get_top_k_concepts(self) -> list:
        return self.top_k_concepts


def completer(text: str, state: int) -> str:
    options = [i for i in completer.options if i.startswith(text)]
    if state < len(options):
        return options[state]
    else:
        return ""


def handle_choice(
    choice: str, node: CustomNode, concept_manager: ConceptManager
) -> bool:
    if choice == "back":
        return False
    for child in node.children:
        if child.name == choice:
            concept_manager.explore_node(child)
            return True
    print("Invalid choice. Please try again.")
    return True


def explore_hierarchy(concept_manager: ConceptManager) -> None:
    node = concept_manager.root_node
    print(f"Node: {node.name}")

    concept_manager.explore_node(node)

    top_k_concepts = concept_manager.get_top_k_concepts()

    for item, score in top_k_concepts:
        print(f"  {item}: {score:.2f}")

    # Sort children by relevance score
    sorted_children = sorted(
        node.children,
        key=lambda n: concept_manager.similarity_scores.get(n.name, 0),
        reverse=True,
    )
    completer.options = [child.name for child in sorted_children]
    readline.set_completer(completer)
    readline.parse_and_bind("tab: complete")

    try:
        while True:
            choice = input(
                "\nEnter a child node to explore, or 'back' to go up: "
            ).strip()
            if not handle_choice(choice, node, concept_manager):
                break
    except KeyboardInterrupt:
        print("\nExiting exploration...")


if __name__ == "__main__":
    root_directory = "."
    concept_manager = ConceptManager(
        "chatbot AI conversation", root_directory=".")
    explore_hierarchy(concept_manager)
