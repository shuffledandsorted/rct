class User:
    """A quantum dialogue system that responds to natural language and observes the screen"""

    def __init__(self, contracts, screen_capture):
        self._contracts = contracts
        self._screen = screen_capture
        self._watching = False

    @property
    def prompt(self) -> str:
        """Status: [N agents | coherence] showing active agents and system coherence"""
        active = len(self._contracts) * 2
        coherence = self._calculate_coherence()
        watching = "👁️ " if self._watching else ""
        return f"{watching}[{active} agents | {coherence:.2f} coherence] > "

    def say(self, text: str) -> None:
        """Communicate with the system using natural language"""

    def watch(self, enable: bool = True) -> None:
        """Start or stop watching the screen for visual input"""

    def exit(self) -> None:
        """Exit the dialogue, cleaning up any active processes"""

    def run(self):
        """Start the dialogue loop"""
        # Get all public methods (commands) through introspection
        commands = {
            name: method
            for name, method in inspect.getmembers(self, inspect.ismethod)
            if not name.startswith("_")
        }

        while True:
            try:
                text = input(self.prompt).strip()

                # Special case for help
                if text == "help":
                    for name, method in commands.items():
                        print(f"{name}: {method.__doc__}")
                    continue

                # Try to match command
                if text in commands:
                    commands[text]()
                    continue

                # Default to say
                self.say(text)

            except KeyboardInterrupt:
                self.exit()
                break
