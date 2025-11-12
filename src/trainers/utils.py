import numpy as np

# =========================================================
# --- Traffic Scheduling ---
# =========================================================

class CurriculumState:
    def __init__(self, cfg):
        self.cfg = cfg
        self.last_num_cars = 0
        self.max_cars = cfg.max_vehicles
        self.start_return = 15
        self.max_return = 75

    def vehicle_schedule(self, mean_return):
        """Compute number of vehicles based on curriculum config."""
        if self.cfg.traffic_enabled: 
            if self.cfg.curriculum_enabled:
                progress = np.clip((mean_return - self.start_return) / (self.max_return - self.start_return), 0, 1)
                target_cars = int(progress * self.max_cars)

                # hysteresis: adjust slowly (avoid jitter)
                if target_cars > self.last_num_cars:
                    self.last_num_cars += 1
                elif target_cars < self.last_num_cars:
                    self.last_num_cars -= 1

                num = self.last_num_cars

            else:
                num = self.max_cars
        else:
            num = 0

        return num
