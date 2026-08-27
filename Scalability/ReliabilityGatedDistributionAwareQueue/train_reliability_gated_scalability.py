"""Train only the reliability-gated distribution-aware scalability model."""

import argparse
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
if str(PARENT) not in sys.path:
    sys.path.insert(0, str(PARENT))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

# Remove the new model-specific arguments before the preserved pipeline parses
# its original CLI.  They are added back to args so config.json records them.
_custom_parser = argparse.ArgumentParser(add_help=False)
_custom_parser.add_argument("--reliability-tau", type=float, default=4.0)
_custom_parser.add_argument("--encoder-hidden-dim", type=int, default=32)
_custom_args, _remaining = _custom_parser.parse_known_args(sys.argv[1:])
sys.argv = [sys.argv[0]] + _remaining

import train_scalability as pipeline  # noqa: E402
from reliability_gated_model import (  # noqa: E402
    DIAGNOSTIC_NAMES,
    RouteNet_Fermi_ReliabilityGatedDistributionAware,
)


class ConfiguredReliabilityGatedModel(
    RouteNet_Fermi_ReliabilityGatedDistributionAware
):
    def __init__(self, queue_aggregation=None):
        super().__init__(
            encoder_hidden_dim=_custom_args.encoder_hidden_dim,
            reliability_tau=_custom_args.reliability_tau,
            queue_aggregation=queue_aggregation,
        )


def _parse_reliability_args():
    args = _original_parse_args()
    args.aggregation = "reliability_gated_distribution_aware"
    args.reliability_tau = float(_custom_args.reliability_tau)
    args.encoder_hidden_dim = int(_custom_args.encoder_hidden_dim)
    return args


_original_parse_args = pipeline.parse_args
pipeline.parse_args = _parse_reliability_args
pipeline.RouteNet_Fermi = ConfiguredReliabilityGatedModel
pipeline.AGGREGATION_NAMES = DIAGNOSTIC_NAMES


if __name__ == "__main__":
    pipeline.main()
