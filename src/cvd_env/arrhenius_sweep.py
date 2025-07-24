"""
| Step                      | Value Choice                  | Why                                | Next Level                | # pylint: disable=line-too-long
| ------------------------- | ----------------------------- | ---------------------------------- | ------------------------- | # pylint: disable=line-too-long
| Initial sweep             | Arbitrary/plausible values    | Test RL & env under varied regimes | Real values from NIST     | # pylint: disable=line-too-long
| Realistic sweep           | NIST/literature values        | Physics grounding, validation      | Fit to fab/process data   | # pylint: disable=line-too-long
| Sweep as env gets complex | Realistic values, more layers | Test RL under realistic complexity | See if agent still learns | # pylint: disable=line-too-long

These values are “toy” for now—perfect for RL/prototyping.

Should use NIST/literature values as you build up realism.

Sweeping is the right way to probe model/agent sensitivity, both now and later.

As the CVDReactorEnv grows, so does the value of these sweeps.

Reaction:

SiH₄ → SiH₂ + H₂

SiH₄ → Si + 2H₂

Kinetics reference (for "real" mode):
Newman, C.G.; O'Neal, H.E.; Ring, M.A.; Leska, F.; Shipley, N.
Kinetics and mechanism of the silane decomposition,
Int. J. Chem. Kinet. 11, 1167 (1979).
NIST Chemical Kinetics Database:
https://kinetics.nist.gov/kinetics/Detail?id=1979NEW/ONE1167:4

Future use with real fab data:

If you have experimental fab data (e.g., measured temperature/flow profiles and
resulting film thickness), use this script to benchmark and stress-test RL and
simulation robustness:

1. Parameter Fitting (Calibration):
   - Fit the kinetic parameters (k0, Ea) in CVDReactorEnv to minimize the error between
     simulated and observed fab outcomes.
   - This can be done via grid search or optimization routines (e.g., scipy.optimize).
   - The fitted parameters then become the new "center" for k0_list and Ea_list in your
     sweep.

2. Sensitivity & Robustness:
   - After calibration, sweep around the fitted/fab values to test if RL
     policies are robust to model uncertainties.
   - This helps validate whether the RL agent generalizes well and is reliable for
     practical deployment.
"""

from typing import TypedDict

from agents.dqn_trainer import train_dqn
from cvd_env.reactor_env import CVDReactorEnv
from utils.logger import get_logger, setup_logging

logger = get_logger(__name__)


class SweepResult(TypedDict):
    """Type definition for Arrhenius sweep result."""

    k0: float
    Ea: float
    alpha: float
    error: float
    final_T: float
    final_F: float


def run_arrhenius_sweep(
    k0_list: list[float], Ea_list: list[int], alpha_list: list[float]
) -> list[SweepResult]:
    """Run a sweep over Arrhenius parameters for the CVD reactor environment."""
    logger.info("Starting Arrhenius parameter sweep...")
    results = []
    for k0 in k0_list:
        for Ea in Ea_list:
            for alpha in alpha_list:
                logger.info("Running sweep: k0=%s, Ea=%s, alpha=%s", k0, Ea, alpha)
                env = CVDReactorEnv(
                    target_thickness=100.0,
                    max_steps=50,
                    k0=k0,
                    Ea=Ea,
                    alpha=alpha,
                    mode="real",
                    P=1e5,  # Pressure in Pa (1 bar)
                    V=1e-3,  # Volume in m^3 (1 liter)
                )
                model = train_dqn(env, total_timesteps=5000)
                obs, _ = env.reset()
                done = False
                while not done:
                    action, _ = model.predict(obs)
                    obs, _, terminated, truncated, info = env.step(int(action))
                    done = terminated or truncated
                thickness_error = info["thickness_error"]
                results.append(
                    {
                        "k0": k0,
                        "Ea": Ea,
                        "alpha": alpha,
                        "error": thickness_error,
                        "final_T": obs[1],
                        "final_F": obs[2],
                    }
                )
    return results


if __name__ == "__main__":
    setup_logging()

    _k0_list = [1e14, 3.16e15, 1e16]  # s^-1
    _Ea_list = [230000, 237794, 250000]  # J/mol
    _alpha_list = [1.0]  # First order
    sweep_results = run_arrhenius_sweep(_k0_list, _Ea_list, _alpha_list)

    logger.info("\n==== Arrhenius Sweep Results ====")
    logger.info(
        "%10s  %10s  %7s  %12s  %10s  %14s",
        "k0 (s^-1)",
        "Ea (J/mol)",
        "alpha",
        "error (nm)",
        "final_T (K)",
        "final_F (sccm)",
    )
    for r in sweep_results:
        logger.info(
            "%10.2e  %10.0f  %7.2f  %12.2f  %10.2f  %14.2f",
            r["k0"],
            r["Ea"],
            r["alpha"],
            r["error"],
            r["final_T"],
            r["final_F"],
        )
