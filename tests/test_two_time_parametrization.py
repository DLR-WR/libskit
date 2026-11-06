"""Coupled two-time coordinates and their physical-to-internal gradients."""

import itertools

import nlopt
import numpy as np
import pytest

import libskit
from libskit._parametrization import TwoTimeLogSoftmaxALRParametrization


NAMES = [
    "early_temperature", "early_electron_density", "early_length",
    "delta_temperature", "delta_electron_density", "delta_length",
    "weight_early", "Fe", "Ca", "scale_factor",
]
INITIAL = np.array([10000., 1e23, 1e-4, 5000., 5e22, 5e-5, .2, .5, .5, 2.])
LOWER = np.array([5000., 1e22, 1e-7, 1., 1e21, 1e-6, 0., 1e-8, 1e-8, .01])
UPPER = np.array([50000., 1e26, .1, 45000., 1e24, 1e-4, 1., 1., 1., 100.])


def make_strategy(initial=None, lower=None, upper=None):
    strategy = TwoTimeLogSoftmaxALRParametrization()
    strategy.bind(
        param_names=NAMES,
        initial_physical=INITIAL if initial is None else initial,
        lower_physical=LOWER if lower is None else lower,
        upper_physical=UPPER if upper is None else upper,
        element_mask=np.array([name in ("Fe", "Ca") for name in NAMES]),
    )
    return strategy


def test_round_trip_accepts_large_feasible_delta():
    strategy = make_strategy()
    physical = strategy.physical_from_internal(strategy.initial_internal)
    np.testing.assert_allclose(physical, INITIAL)
    assert physical[1] - physical[4] == pytest.approx(5e22)


def test_all_bound_corners_are_physical():
    strategy = make_strategy()
    for endpoints in itertools.product((0, 1), repeat=4):
        x = strategy.initial_internal
        for index, endpoint in zip((0, 1, 3, 4), endpoints):
            x[index] = (strategy.lower_internal, strategy.upper_internal)[endpoint][index]
        physical = strategy.physical_from_internal(x)
        assert physical[0] > physical[3] > 0
        assert physical[1] > physical[4] > 0
        assert physical[0] - physical[3] >= 1000 * (1 - 1e-12)
        assert physical[1] - physical[4] >= 1e18 * (1 - 1e-6)
        for index in (3, 4):
            assert LOWER[index] <= physical[index] <= UPPER[index] * (1 + 1e-14)
        assert physical[7:9].sum() == pytest.approx(1)


@pytest.mark.parametrize("early_temperature", [10000., 49000.])
@pytest.mark.parametrize("early_density", [1e23, 2e24])
def test_chain_rule_matches_finite_differences(early_temperature, early_density):
    strategy = make_strategy()
    internal = strategy.initial_internal
    internal[0] = np.log(early_temperature)
    internal[1] = np.log(early_density)
    internal[3:5] = [.4, .6]
    internal[7] = .7
    weights = np.linspace(.7, 1.6, len(NAMES))
    scales = INITIAL

    def objective(x):
        physical = strategy.physical_from_internal(x)
        return np.sum(weights * (physical / scales) ** 2)

    physical = strategy.physical_from_internal(internal)
    gradient = 2 * weights * physical / scales ** 2
    actual = strategy.chain_rule(dict(zip(NAMES, gradient)), physical)
    for i in range(len(NAMES)):
        if strategy.lower_internal[i] == strategy.upper_internal[i]:
            assert actual[i] == 0
            continue
        plus, minus = internal.copy(), internal.copy()
        plus[i] += 1e-5
        minus[i] -= 1e-5
        expected = (objective(plus) - objective(minus)) / 2e-5
        assert actual[i] == pytest.approx(expected, rel=2e-5, abs=1e-5)


@pytest.mark.parametrize(
    "index,value",
    [(0, 4000.), (3, 10000.), (4, 1e23), (4, np.nan)],
)
def test_invalid_initial_states_fail(index, value):
    initial = INITIAL.copy()
    initial[index] = value
    with pytest.raises(ValueError):
        make_strategy(initial=initial)


def test_infeasible_bounds_fail():
    lower = LOWER.copy()
    lower[4] = 2e22
    with pytest.raises(ValueError, match="lower_delta"):
        make_strategy(lower=lower)


def test_non_two_time_model_fails():
    strategy = TwoTimeLogSoftmaxALRParametrization()
    with pytest.raises(ValueError, match="requires early_temperature"):
        strategy.bind(
            param_names=["temperature"],
            initial_physical=np.array([10000.]),
            lower_physical=np.array([5000.]),
            upper_physical=np.array([50000.]),
            element_mask=np.array([False]),
        )


@pytest.mark.parametrize("algorithm", [nlopt.LD_LBFGS, nlopt.LD_SLSQP])
def test_optimizer_gradients_and_fit_remain_physical(algorithm):
    wavelengths = np.arange(390., 400., .05)
    instrument = libskit.SlopingSpectrometer(9.0705e-05, 3.2962e-02)
    true_model = libskit.LTETwoTime(
        {"Fe": .7, "Ca": .3}, 12000., 2e23, 1e-4, 5000., 1e23, 5e-5, .4,
    )
    spectrum = libskit.Spectrum(
        wavelengths, instrument.calculate_spectrum(wavelengths, true_model),
    )
    model = libskit.LTETwoTime(
        {"Fe": .5, "Ca": .5}, *INITIAL[:7],
    )
    experiment = libskit.Experiment(
        libskit.LossType.L2Loss, model, instrument, spectrum,
        libskit.NormalizationType.AnalyticNorm,
    )
    bounds = dict(zip(NAMES, zip(LOWER, UPPER)))
    optimizer = libskit.Optimizer(
        experiment, bounds=bounds,
        parametrization=libskit.Parametrization.LOG_SOFTMAX_ALR_TWO_TIME,
    )
    x = optimizer._initial_internal.copy()
    optimizer.history = {name: [] for name in optimizer.param_names}
    optimizer.history.update(loss=[], iteration=[])
    gradient = np.empty_like(x)
    initial_loss = optimizer._objective(x, gradient)
    for name in ("early_temperature", "early_electron_density",
                 "delta_temperature", "delta_electron_density"):
        index = optimizer.param_names.index(name)
        plus, minus = x.copy(), x.copy()
        plus[index] += 1e-5
        minus[index] -= 1e-5
        expected = (
            optimizer._objective_no_grad(plus)
            - optimizer._objective_no_grad(minus)
        ) / 2e-5
        assert gradient[index] == pytest.approx(expected, rel=1e-3, abs=1e-7)
    optimizer._apply_params(x)
    result = optimizer.optimize(algorithm=algorithm, max_eval=60)
    assert result["status"] == "success", result
    assert result["final_loss"] < initial_loss
    history = result["history"]
    for quantity in ("temperature", "electron_density"):
        early = np.asarray(history[f"early_{quantity}"])
        delta = np.asarray(history[f"delta_{quantity}"])
        assert np.isfinite(early - delta).all()
        assert (early > delta).all()
