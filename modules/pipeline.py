from modules.dataset import Dataset
import pandas as pd
import time
from scipy.optimize import minimize, differential_evolution
from scipy.optimize import NonlinearConstraint
from functools import partial


from modules.eval import (
    compute_calibration_errors,
)
from modules.optimization import (
    loss_pipeline,
    parameterization_to_transform,
    parameterize_transformation,
)


import numpy as np
import quaternion  # NOTE: use this instead of scipy bec. it has built in normalization of quat


def unit_quaternion_constraint(p):
    q = p[:4]
    return np.sqrt(np.sum(q**2))


class Pipeline:
    def __init__(self, dataset: Dataset, optimizer_tolerance: float):
        if isinstance(dataset.points, list):
            raise TypeError(
                f"Expected dataset.points to be of type np.ndarray, but got list. Please preprocess your "
                "the point clouds in your dataset before pipeline initialization."
            )

        if len(dataset.points.shape) < 3 or dataset.points.shape[1] != 4:
            raise ValueError(
                f"Expected dataset.points to have shape (B, 4, N), "
                "but got shape {dataset.points.shape}. Please preprocess your "
                "the point clouds in your dataset before pipeline initialization."
            )

        self.dataset = dataset
        self.optimizer_tolerance = optimizer_tolerance
        # Properties set after last optimization run.
        self.T_star = None  # extrinsic calibration optimizer converged to
        self.processing_time = None  # time last self.optimize run took
        self.optimizer_log = None

    def optimize(
        self,
        T_init: np.ndarray = None,
        rotation_only: bool = False,
        num_bins=128,
    ):
        """Optimizes from ground truth perturbed with given error magnitudes and saves results as properties."""
        start_t = time.time()
        try:
            # rotation-only parameterization
            p0, t, Rp = parameterize_transformation(
                T=T_init,
                rotation_only=True
            )

            # rotation-only optimization
            res = minimize(
                loss_pipeline,
                p0,
                args=(
                    self.dataset.points,
                    self.dataset.P,
                    self.dataset.camera_depths,
                    self.dataset.camera_masks,
                    (self.dataset.image_height, self.dataset.image_width),
                    t,
                    Rp,
                    num_bins,
                ),
                method="Nelder-Mead",
                # options={"maxfev": 3200},
                tol=self.optimizer_tolerance,
            )
            self.set_solution(res.x, t, R_mat=Rp)

            # DE (joint) optimization
            if not rotation_only:
                p0, t, Rp = parameterize_transformation(
                    T=self.T_star,
                    rotation_only=False
                )

                # limits search space using bounds
                t_bound_radius = 0.5  # 50 cm
                q_bounds_radius = (
                    0.02  # quaternion bounds (also assumes < 5 deg initial rot error)
                )

                # force unit norm of quat.
                nlc1 = NonlinearConstraint(unit_quaternion_constraint, 0.999, 1.001)
                # force rotation magnitude from initial matrix to be less than 5 deg
                nlc2 = None

                def bound_rotation_magnitude(p, R_init: np.ndarray):
                    R_proposed = quaternion.as_rotation_matrix(
                        np.quaternion(p[0], p[1], p[2], p[3])
                    )
                    magnitude = compute_calibration_errors(
                        T_pred=R_proposed,
                        T_gt=R_init,
                        log=True,
                        only_rot_magnitude=True,
                    )
                    return np.abs(magnitude)

                bound_rotation_magnitude_with_fixed_T_init = partial(
                    bound_rotation_magnitude, R_init=self.T_star[:3, :3].copy()
                )
                nlc2 = NonlinearConstraint(
                    bound_rotation_magnitude_with_fixed_T_init, 0, 6
                )
                de_options = {
                    "constraints": [nlc1, nlc2],
                    "init": "latinhypercube",
                    "maxiter": 1000,
                    "popsize": 20,
                    "disp": True,
                    "strategy": "best1bin",
                    "tol": self.optimizer_tolerance,
                    "atol": self.optimizer_tolerance,
                    "updating": "deferred",
                    "workers": 5,
                    "polish": True,        # polish w/ 'trust-constr'
                    "mutation": 1.2,
                    "recombination": 0.6,
                }
                center_values = p0.copy()  # with p0 = [q1, q2, q3, q4, tx, ty, tz]
                q_bounds = [
                    (
                        np.clip(center_values[i] - q_bounds_radius, -1, +1),
                        np.clip(center_values[i] + q_bounds_radius, -1, +1),
                    )
                    for i in range(4)
                ]
                t_bounds = [
                    (
                        center_values[i] - t_bound_radius,
                        center_values[i] + t_bound_radius,
                    )
                    for i in range(4, 4 + 3)
                ]
                de_bounds = q_bounds + t_bounds
                res = differential_evolution(
                    func=loss_pipeline,
                    bounds=de_bounds,  # Use the defined bounds
                    args=(
                        self.dataset.points,
                        self.dataset.P,
                        self.dataset.camera_depths,
                        self.dataset.camera_masks,
                        (self.dataset.image_height, self.dataset.image_width),
                        t,
                        Rp,
                        num_bins,
                    ),
                    **de_options,
                )
            self.optimizer_log = res

            # save found solution
            p = res.x
            self.set_solution(p, t, R_mat=Rp)
            self.processing_time = time.time() - start_t
        except Exception as e:
            raise e
            # print(
            #    f"[WARNING] Optimization failed. The following exception was raised: {e}"
            # )



    def set_solution(self, p: np.ndarray, t: np.ndarray, R_mat: np.ndarray ) -> np.ndarray:
        if R_mat is not None:
            self.T_star = parameterization_to_transform(p=p, R_mat=R_mat)
        else:
            if t is not None:
                p = np.concatenate((p, t))
            self.T_star = parameterization_to_transform(
                p=p,
            )



    def evaluate(self) -> dict:
        """Prints the
            - optimizer's output
            - processing time
            - calibration errors

        Returns:
            a dictionary containing the tracked metrics.
        """
        if self.T_star is None:
            raise Exception("No optimization run has finished.")

        print("Optim. Output:", self.optimizer_log)
        print()
        print("Time:", self.processing_time)
        print()
        (
            r_err_magnitude,
            roll_err,
            pitch_err,
            yaw_err,
            t_err_magnitude,
            tx_err,
            ty_err,
            tz_err,
        ) = compute_calibration_errors(
            T_pred=self.T_star, T_gt=self.dataset.T_gt, log=True
        )
        return {
            "r_err_magnitude": r_err_magnitude,
            "roll_err": roll_err,
            "pitch_err": pitch_err,
            "yaw_err": yaw_err,
            "t_err_magnitude": t_err_magnitude,
            "tx_err": tx_err,
            "ty_err": ty_err,
            "tz_err": tz_err,
            "time": self.processing_time,
            "nid": np.round(self.optimizer_log.fun, decimals=3),
        }
