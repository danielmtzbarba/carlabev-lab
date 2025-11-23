import numpy as np

# =========================================================
# --- Traffic Scheduling ---
# =========================================================


class CurriculumState:
    def __init__(self, cfg):
        self.cfg = cfg
        self.last_num_cars = 0
        self.max_cars = cfg.max_vehicles
        self.curr_mode = cfg.curriculum_mode
        self.curr_veh = self.curr_mode in ["vehicles_only", "both"]
        self.curr_route = self.curr_mode in ["route_only", "both"]
        self.start_return = 15
        self.max_return = 50

        self.min_dist_start = 30  # max route length in meters
        self.max_dist_target = 1000  # max route length in meters
        self.range_width_start = 100  # initial max-min distance (100 - 30)
        self.range_width_target = 400  # target range width for sampling

        self.last_min = 30
        self.last_width = 100

    def vehicle_schedule(self, mean_return):
        """Adaptive number of vehicles based on curriculum difficulty."""

        # Traffic OFF globally → always 0
        if not self.cfg.traffic_enabled:
            return 0

        # Curriculum fully disabled → always max
        if not self.cfg.curriculum_enabled:
            return self.max_cars

        # Curriculum does not apply to vehicles → always 0
        if not self.curr_veh:
            return self.max_cars

        # -------------------------------------------------
        # 1. Compute target difficulty level (0 → 1)
        # -------------------------------------------------
        progress = np.clip(
            (mean_return - self.start_return) / (self.max_return - self.start_return),
            0.0,
            1.0,
        )

        target_cars = int(progress * self.max_cars)  # desired level

        # -------------------------------------------------
        # 2. Hysteresis: smooth both increasing AND decreasing
        # -------------------------------------------------
        if target_cars > self.last_num_cars:
            self.last_num_cars += 1  # ramp up slowly
        elif target_cars < self.last_num_cars:
            self.last_num_cars -= 1  # ramp down slowly

        # -------------------------------------------------
        # 3. Clip and return
        # -------------------------------------------------
        self.last_num_cars = int(np.clip(self.last_num_cars, 0, self.max_cars))
        return self.last_num_cars

    def route_schedule(self, mean_return):
        """Return a range [min_dist, max_dist] for route sampling based on curriculum."""

        # If curriculum disabled → full traffic from start
        if not self.cfg.curriculum_enabled:
            return [self.min_dist_start, self.max_dist_target]

        # If curriculum mode does NOT include vehicles → always 0
        if not self.curr_route:
            return [self.min_dist_start, self.max_dist_target]

        # Normalize return for progress (0-1)
        progress = np.clip(
            (mean_return - self.start_return) / (self.max_return - self.start_return),
            0,
            1,
        )

        # Compute target values
        target_min_dist = int(
            self.min_dist_start
            + progress * (self.max_dist_target - self.min_dist_start)
        )
        target_width = int(
            self.range_width_start
            + progress * (self.range_width_target - self.range_width_start)
        )

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
        self.last_min = np.clip(
            self.last_min, self.min_dist_start, self.max_dist_target
        )
        self.last_width = min(self.last_width, self.max_dist_target - self.last_min)

        return [self.last_min, self.last_min + self.last_width]
