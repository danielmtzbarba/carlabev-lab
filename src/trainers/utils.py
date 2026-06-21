import numpy as np

# =========================================================
# --- Traffic Scheduling ---
# =========================================================


class CurriculumState:
    def __init__(self, protocol):
        self.protocol = protocol
        self.backbone = protocol.backbone
        self.last_num_cars = 0
        self.max_cars = int(
            self.backbone.num_vehicles_near_ego
            if self.backbone.num_vehicles_near_ego is not None
            else (self.backbone.num_vehicles or 0)
        )
        self.curr_axis = protocol.curriculum_axis
        self.curr_veh = self.curr_axis in ["near_ego_traffic", "both"]
        self.curr_route = self.curr_axis in ["route_distance", "both"]
        self.start_return = 15
        self.max_return = 50

        initial_range = self.backbone.route_dist_range or (30, 100)
        self.min_dist_start = int(initial_range[0])
        self.range_width_start = int(initial_range[1] - initial_range[0])
        self.max_dist_target = 1000
        self.range_width_target = 400

        self.last_min = self.min_dist_start
        self.last_width = self.range_width_start

    def vehicle_schedule(self, mean_return):
        """Adaptive number of vehicles with asymmetric hysteresis:
        - increases slowly
        - decreases rapidly when performance drops
        """

        # No traffic globally
        if self.max_cars <= 0:
            return 0

        # Curriculum disabled
        if not self.protocol.use_curriculum or not self.curr_veh:
            return self.max_cars

        # ======================================================
        # 1. Compute normalized difficulty signal (0 → 1)
        # ======================================================
        progress = np.clip(
            (mean_return - self.start_return) / (self.max_return - self.start_return),
            0.0,
            1.0,
        )

        target_cars = int(progress * self.max_cars)

        # ======================================================
        # 2. Asymmetric progression
        #    - slow increase
        #    - fast decrease
        # ======================================================
        if target_cars > self.last_num_cars:
            # slow ramp-up (reduce oscillation)
            self.last_num_cars += 1

        elif target_cars < self.last_num_cars:
            # fast ramp-down (recover when overwhelmed)
            self.last_num_cars -= 3

        # Clip to valid range
        self.last_num_cars = int(np.clip(self.last_num_cars, 0, self.max_cars))
        return self.last_num_cars

    def route_schedule(self, mean_return):
        """Return a range [min_dist, max_dist] for route sampling based on curriculum."""

        # If curriculum disabled → full traffic from start
        if not self.protocol.use_curriculum:
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
