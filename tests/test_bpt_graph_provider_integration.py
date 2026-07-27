from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from causal4d.graph_provider_contract import (
    BAYESIAN_PHYSTWIN_GRAPH_PARENT_PROVIDER_API,
    BAYESIAN_PHYSTWIN_GRAPH_PARENT_PROVIDER_API_VERSION,
    BAYESIAN_PHYSTWIN_GRAPH_PROVIDER_API,
    require_bayesian_phystwin_graph_provider,
)


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("bayesian_phystwin") is None,
    reason="Bayesian-PhysTwin provider is not installed",
)


def test_installed_graph_provider_builds_and_groups() -> None:
    from bayesian_phystwin.causal4d_graph_provider_v1 import (
        PhysTwinSpringGraphConfig,
        build_phystwin_spring_graph,
        controller_hand_count,
        infer_controller_groups,
    )

    manifest = require_bayesian_phystwin_graph_provider(provider_revision="b" * 40)
    assert manifest.metadata["provider_api"] == BAYESIAN_PHYSTWIN_GRAPH_PROVIDER_API
    assert manifest.metadata["parent_provider_api"] == (
        BAYESIAN_PHYSTWIN_GRAPH_PARENT_PROVIDER_API
    )
    assert manifest.metadata["parent_provider_api_version"] == (
        BAYESIAN_PHYSTWIN_GRAPH_PARENT_PROVIDER_API_VERSION
    )

    controls = np.asarray(
        [
            [-0.2, 0.0, 0.0],
            [-0.1, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [0.2, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    assert controller_hand_count("double_stretch_sloth") == 2
    np.testing.assert_array_equal(
        infer_controller_groups(controls, group_count=2),
        np.asarray([0, 0, 1, 1], dtype=np.int32),
    )

    graph = build_phystwin_spring_graph(
        np.asarray(
            [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0]],
            dtype=np.float32,
        ),
        controls[:1],
        config=PhysTwinSpringGraphConfig(
            object_radius=0.15,
            object_max_neighbours=3,
            controller_radius=0.25,
            controller_max_neighbours=2,
        ),
    )
    assert graph.num_object_points == 3
    assert graph.springs.shape[1] == 2
