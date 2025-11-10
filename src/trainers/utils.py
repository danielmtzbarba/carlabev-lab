import numpy as np

self.curriculum_enabled = cfg.get("curriculum_enabled", True)
self.start_ep = cfg.get("start_ep", 1)
self.max_v = cfg.get("max_vehicles", 50)
self.mid = cfg.get("midpoint", 10)
self.growth_rate = cfg.get("growth_rate", 0.01)

# =========================================================
# --- Traffic Scheduling ---
# =========================================================
def _vehicle_schedule(cfg):
    """Compute number of vehicles based on curriculum config."""
    if episode < self.start_ep:
        return 0

    # Logistic curve growth
    num = int(self.max_v / (1 + np.exp(-self.growth_rate * (episode - self.mid))))
    # Add stochastic jitter
    return int(np.clip(num + np.random.randint(-3, 4), 0, self.max_v))
