import numpy as np
import mss
import time
import threading
from typing import Optional, Tuple
from .flow import FlowAgent
from .config import AgentConfig
from scipy.ndimage import zoom


class ScreenCaptureAgent(FlowAgent):
    """Agent that captures screen and processes it through quantum states"""

    def __init__(
        self, dims: Tuple[int, int], config: Optional[AgentConfig] = None, age: int = 0
    ):
        super().__init__(dims=dims, config=config, age=age)
        self.sct = mss.mss()
        self.is_capturing = False
        self.capture_thread: Optional[threading.Thread] = None
        self.capture_interval = 2.0  # seconds between captures
        self.last_status_time = 0
        self.status_interval = 2.0  # Print status every 2 seconds
        self.interested_agents = 0  # Count of agents interested in screen state
        self.total_agents = 0  # Total number of agents

        # Find primary monitor
        self.monitor = self.sct.monitors[0]  # Primary monitor
        print(f"[SCREEN] Found {len(self.sct.monitors)} monitor(s)")

    def start_capture(self):
        """Start continuous screen capture in a separate thread"""
        if self.is_capturing:
            return

        self.is_capturing = True
        self.capture_thread = threading.Thread(target=self._capture_loop)
        self.capture_thread.daemon = True
        print("[SCREEN] Starting capture...", end="", flush=True)
        self.capture_thread.start()

    def stop_capture(self):
        """Stop the screen capture thread"""
        self.is_capturing = False
        if self.capture_thread:
            self.capture_thread.join()
            self.capture_thread = None
            print("\r[SCREEN] Capture stopped" + " " * 50)

    def update_agent_counts(self, total: int, interested: int):
        """Update the count of total and interested agents"""
        self.total_agents = total
        self.interested_agents = interested

    def _capture_loop(self):
        """Main capture loop that runs in a separate thread"""
        last_status = ""
        while self.is_capturing:
            try:
                current_time = time.time()

                # Capture the primary monitor
                screen = self.sct.grab(self.monitor)

                # Convert to numpy array and normalize
                img = np.array(screen)

                if img.size == 0:
                    new_status = "[SCREEN] Error: Empty screen"
                    if new_status != last_status:
                        print(f"\n{new_status}")
                        print("You: ", end="", flush=True)
                        last_status = new_status
                    time.sleep(1)
                    continue

                # Convert to grayscale for quantum state processing
                gray = np.mean(img, axis=2)

                # Resize to match agent dimensions
                h, w = self.dims
                resized = self._resize_image(gray, (h, w))

                # Update quantum state with the new image
                self.update_quantum_state(resized)

                # Print periodic status updates
                if current_time - self.last_status_time >= self.status_interval:
                    if hasattr(self.wave_fn, "amplitude"):
                        # Calculate some basic statistics about what we're seeing
                        amplitude = np.abs(self.wave_fn.amplitude)
                        avg_intensity = np.mean(amplitude)
                        activity = np.std(amplitude)

                        # Single line status with activity indicator
                        activity_bar = "▁▂▃▄▅▆▇█"
                        activity_level = min(int(activity * 8), 7)
                        activity_indicator = activity_bar[activity_level]

                        # Enhanced status with agent counts
                        agent_info = (
                            f"[{self.interested_agents}/{self.total_agents}]"
                            if self.total_agents > 0
                            else ""
                        )
                        new_status = f"[SCREEN] {activity_indicator} i={avg_intensity:.2f} a={activity:.2f} {agent_info}"
                        if new_status != last_status:
                            print(f"\n{new_status}")
                            print("You: ", end="", flush=True)
                            last_status = new_status

                    self.last_status_time = current_time

                # Wait for next capture
                time.sleep(self.capture_interval)

            except Exception as e:
                new_status = f"[SCREEN] Error: {str(e)[:50]}"
                if new_status != last_status:
                    print(f"\n{new_status}")
                    print("You: ", end="", flush=True)
                    last_status = new_status
                time.sleep(1)  # Wait before retrying

    def _resize_image(
        self, img: np.ndarray, target_size: Tuple[int, int]
    ) -> np.ndarray:
        """Resize image to target dimensions using scipy zoom for better interpolation"""
        h, w = target_size
        zoom_h = h / img.shape[0]
        zoom_w = w / img.shape[1]

        # Use scipy zoom with order=1 for bilinear interpolation
        resized = zoom(img, (zoom_h, zoom_w), order=1)

        # Ensure output is normalized
        resized = (resized - resized.min()) / (resized.max() - resized.min() + 1e-10)

        return resized
