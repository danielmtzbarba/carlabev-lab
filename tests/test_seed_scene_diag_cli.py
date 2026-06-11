from __future__ import annotations

import argparse
import unittest

from scripts.analyze_seed_scene_distribution import (
    _first_n_primes,
    _resolve_intersection_required,
    _resolve_route_profile_mix,
    _resolve_seeds,
)


class SeedSceneDiagCliTests(unittest.TestCase):
    def test_first_n_primes_generates_expected_sequence(self):
        self.assertEqual(_first_n_primes(10), [2, 3, 5, 7, 11, 13, 17, 19, 23, 29])

    def test_resolve_seeds_prefers_prime_generation_when_requested(self):
        args = argparse.Namespace(seeds=None, prime_seeds_count=5, prime_seeds_start=10)
        self.assertEqual(_resolve_seeds(args), [11, 13, 17, 19, 23])

    def test_resolve_seeds_rejects_mixed_seed_sources(self):
        args = argparse.Namespace(seeds=[2, 3], prime_seeds_count=2, prime_seeds_start=2)
        with self.assertRaises(ValueError):
            _resolve_seeds(args)

    def test_resolve_route_profile_mix_parses_weights(self):
        args = argparse.Namespace(route_profile_mix=["mostly_straight=0.4", "single_left=0.6"])
        self.assertEqual(
            _resolve_route_profile_mix(args),
            {"mostly_straight": 0.4, "single_left": 0.6},
        )

    def test_resolve_intersection_required_rejects_conflict(self):
        args = argparse.Namespace(intersection_required=True, intersection_forbidden=True)
        with self.assertRaises(ValueError):
            _resolve_intersection_required(args)

    def test_default_ego_route_graph_is_full_vehicle(self):
        args = argparse.Namespace(ego_route_graph="full_vehicle")
        self.assertEqual(args.ego_route_graph, "full_vehicle")


if __name__ == "__main__":
    unittest.main()
