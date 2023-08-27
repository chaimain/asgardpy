import os

import pytest


@pytest.mark.test_data
def test_dataset1d_full(base_config):
    from asgardpy.analysis import AsgardpyAnalysis

    analysis = AsgardpyAnalysis(base_config)

    analysis.run(["datasets-1d"])
