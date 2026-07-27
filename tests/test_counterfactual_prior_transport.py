import numpy as np
import pytest

from causal4d.contracts import (
    ActionWindow,
    CausalContext,
    FactualIntervention,
    ObservationWindow,
)
from causal4d.counterfactual import (
    _new_contact_weights,
    _same_grasp_weights,
    _same_patch_weights,
)
from causal4d.rollout_bank import JointRolloutBank


_DIGEST = "0" * 64


def _context() -> CausalContext:
    return CausalContext(
        protocol_id="counterfactual-prior-transport-test",
        o_minus=ObservationWindow("case", "points", 0, 2, _DIGEST),
        o_plus=ObservationWindow("case", "points", 2, 5, _DIGEST),
        u_obs=ActionWindow("observed", "case", 0, 5, _DIGEST, "recorded"),
        u_cf=ActionWindow("query", "case", 2, 5, _DIGEST, "counterfactual"),
    )


def _factual(
    *,
    phi: tuple[float, float, float] = (1.0, 0.0, 0.0),
    kappa: tuple[float, float] = (0.0, 0.0),
) -> FactualIntervention:
    return FactualIntervention(
        context=_context(),
        component_ids=("factual",),
        phi_names=("gain_multiplier", "delay_steps", "rotation_degrees"),
        kappa_names=("attachment_shift_hand_0", "slip_fraction"),
        phi=np.asarray([phi], dtype=float),
        kappa_obs=np.asarray([kappa], dtype=float),
        hypothesis_indices=np.asarray([0]),
        twin_particle_indices=np.asarray([0]),
        weights=np.asarray([1.0]),
        evidence_frame_stop=3,
        source_twin_belief_id=_DIGEST,
    )


def _metadata(
    identifier: str,
    *,
    gain: float = 1.0,
    shift: int = 0,
    slip: float = 0.0,
) -> dict[str, object]:
    return {
        "hypothesis_id": identifier,
        "contact": {
            "attachment_shifts": [shift],
            "gain_multiplier": gain,
            "delay_steps": 0,
            "slip_fraction": slip,
            "rotation_degrees": 0.0,
        },
    }


def _bank(
    metadata: tuple[dict[str, object], ...],
    priors: tuple[float, ...],
) -> JointRolloutBank:
    return JointRolloutBank(
        hypothesis_ids=tuple(f"h{index}" for index in range(len(metadata))),
        hypothesis_metadata=metadata,
        hypothesis_prior_weights=np.asarray(priors, dtype=float),
        parameter_particles=np.asarray([[0.0]]),
        parameter_weights=np.asarray([1.0]),
        trajectories=np.zeros((len(metadata), 1, 2, 1, 3), dtype=np.float32),
    )


@pytest.mark.parametrize(
    ("priors", "expected"),
    [
        ((0.6, 0.2, 0.2), (0.75, 0.25, 0.0)),
        ((0.8, 0.0, 0.2), (1.0, 0.0, 0.0)),
    ],
)
def test_same_grasp_equivalent_hypotheses_follow_query_prior(
    priors: tuple[float, ...],
    expected: tuple[float, ...],
) -> None:
    bank = _bank(
        (
            _metadata("equivalent-a"),
            _metadata("equivalent-b"),
            _metadata("other-contact", shift=1),
        ),
        priors,
    )

    weights, retained_mass = _same_grasp_weights(bank, _factual())

    assert np.allclose(weights[:, 0], expected)
    assert retained_mass == pytest.approx(1.0)


def test_same_patch_resamples_slip_from_conditional_query_prior() -> None:
    bank = _bank(
        (
            _metadata("low-slip", slip=0.0),
            _metadata("high-slip", slip=0.4),
            _metadata("other-patch", shift=1),
        ),
        (0.45, 0.15, 0.40),
    )

    weights, retained_mass = _same_patch_weights(
        bank,
        _factual(kappa=(0.0, 0.2)),
    )

    assert np.allclose(weights[:, 0], [0.75, 0.25, 0.0])
    assert retained_mass == pytest.approx(1.0)


def test_new_contact_fails_closed_for_zero_prior_phi_support() -> None:
    bank = _bank(
        (
            _metadata("supported", gain=1.0),
            _metadata("zero-prior", gain=1.2),
        ),
        (1.0, 0.0),
    )

    with pytest.raises(ValueError, match="no support for factual phi posterior"):
        _new_contact_weights(bank, _factual(phi=(1.2, 0.0, 0.0)))
