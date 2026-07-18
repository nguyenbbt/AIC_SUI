from __future__ import annotations

import math
import unittest

from pydantic import ValidationError

from online.domain.enums import RetrievalBranch
from online.ranking.fusion import FRAME_FUSION_BRANCHES
from online.ranking.policy import RankingPolicyConfig


class RankingPolicyConfigTests(unittest.TestCase):
    def test_policy_config_is_frozen_and_mappings_are_immutable(self) -> None:
        config = RankingPolicyConfig(query_variant_weights={"q0": 1.0})

        with self.assertRaises(ValidationError):
            config.policy_name = "other"  # type: ignore[misc]
        with self.assertRaises(TypeError):
            config.query_variant_weights["q0"] = 0.5  # type: ignore[index]

    def test_policy_config_rejects_extra_invalid_and_non_finite_values(self) -> None:
        with self.assertRaises(ValidationError):
            RankingPolicyConfig.model_validate({"unknown": "value"})
        with self.assertRaises(ValidationError):
            RankingPolicyConfig(query_variant_weights={"q0": -1.0})
        with self.assertRaises(ValidationError):
            RankingPolicyConfig(summary_weight=2.0)
        with self.assertRaises(ValidationError):
            RankingPolicyConfig(query_variant_weights={"q0": math.inf})
        with self.assertRaises(ValidationError):
            RankingPolicyConfig(
                fusion_default_weight=0.0,
                fusion_weights={RetrievalBranch.VISUAL_DENSE: 0.0},
            )
        with self.assertRaises(ValidationError):
            RankingPolicyConfig(
                fusion_default_weight=1.0,
                fusion_weights={branch: 0.0 for branch in FRAME_FUSION_BRANCHES},
            )
        with self.assertRaises(ValidationError):
            RankingPolicyConfig(
                fusion_weights={RetrievalBranch.SUMMARY_DENSE: 1.0},
            )

if __name__ == "__main__":
    unittest.main()

