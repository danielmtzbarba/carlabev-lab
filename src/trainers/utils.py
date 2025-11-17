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
        self.max_return = 50
        self.min_dist = 30        # meters
        self.max_dist = 1000      # meters
        self.last_route_dist = 100  # starting value for smoothing

        self.max_dist_target = 1000  # max route length in meters
        self.range_width_start = 100  # initial max-min distance (100 - 30)
        self.range_width_target = 400  # target range width for sampling
        
        self.last_min = 30
        self.last_width = 100

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

    def route_schedule(self, mean_return):
        """Return a range [min_dist, max_dist] for route sampling based on curriculum."""

        if not self.cfg.curriculum_enabled:
            return [self.min_dist_start, self.max_dist_target]

        # Normalize return for progress (0-1)
        progress = np.clip(
            (mean_return - self.start_return) / (self.max_return - self.start_return),
            0, 1
        )

        # Compute target values
        target_min_dist = int(self.min_dist_start + progress * (self.max_dist_target - self.min_dist_start))
        target_width = int(self.range_width_start + progress * (self.range_width_target - self.range_width_start))

        # Smooth (hysteresis)
        if target_min_dist > self.last_min:
            self.last_min += 10
        elif target_min_dist < self.last_min:
            self.last_min -= 10

        if target_width > self.last_width:
            self.last_width += 10
        elif target_width < self.last_width:
            self.last_width -= 10

        # Clip to avoid invalid values
        self.last_min = np.clip(self.last_min, self.min_dist_start, self.max_dist_target)
        self.last_width = min(self.last_width, self.max_dist_target - self.last_min)

        return [self.last_min, self.last_min + self.last_width]
