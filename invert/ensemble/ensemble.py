import logging

import numpy as np

from ..invert import Solver

logger = logging.getLogger(__name__)


class Ensemble:
    def __init__(self, solver_list: list, summary_type="median", *args, **kwargs):
        if all(isinstance(solver_list, str) for x in solver_list):
            self.solvers = [
                Solver(solver_name, *args, **kwargs) for solver_name in solver_list
            ]
            self.solver_names = solver_list
        else:
            self.solvers = [Solver(solver_name) for solver_name in solver_list]
            self.solver_names = [solver.name for solver in self.solvers]  # type: ignore[attr-defined]

        self.summary_types = ["mean", "median", "covariance", "likelihood"]
        self.name = "Ensemble"
        self.summary_type = summary_type
        self.n_solvers = len(self.solvers)

    def make_inverse_operator(self, *args, **kwargs):
        for i in range(self.n_solvers):
            self.solvers[i].make_inverse_operator(*args, **kwargs)

    def apply_inverse_operator(self, mne_obj, *args, **kwargs):
        stc_list = []
        data = self.solvers[0].unpack_data_obj(mne_obj)
        self.neg_log_likelihoods = []
        for i in range(self.n_solvers):
            stc_single = self.solvers[i].apply_inverse_operator(
                mne_obj, *args, **kwargs
            )
            self.neg_log_likelihoods.append(
                self.calc_neg_log_likelihood(stc_single.data, data)
            )
            stc_list.append(stc_single)
        self.stc_list = stc_list
        stc = self.summarize_predictions(stc_list, mne_obj)

        final_neg_log_likelihood = self.calc_neg_log_likelihood(stc.data, data)

        logger.info("Neg Log Likelihoods:")
        for solver_name, neg_log_likelihood in zip(
            self.solver_names, self.neg_log_likelihoods
        ):
            logger.info(f"{solver_name}: {neg_log_likelihood}")
        logger.info(f"Final likelihood: {final_neg_log_likelihood}")

        return stc

    def summarize_predictions(self, stc_list, mne_obj):
        stc = stc_list[0].copy()
        if self.summary_type.lower() == "mean":
            stc.data = np.nanmean(
                np.stack([stc_.data for stc_ in stc_list], axis=0), axis=0
            )
        elif self.summary_type.lower() == "median":
            stc.data = np.nanmedian(
                np.stack([stc_.data for stc_ in stc_list], axis=0), axis=0
            )
        elif self.summary_type.lower() == "likelihood":
            stc.data = np.average(
                np.stack([stc_.data for stc_ in stc_list], axis=0),
                axis=0,
                weights=np.log(self.neg_log_likelihoods),
            )
        elif self.summary_type.lower() == "covariance":
            data = self.solvers[0].unpack_data_obj(mne_obj)
            mean_prediction = np.nanmean(
                np.stack([stc_.data for stc_ in stc_list], axis=0), axis=0
            )
            mean_prediction = np.mean(np.abs(mean_prediction), axis=-1)
            source_covariance = np.diag(mean_prediction)
            leadfield = self.solvers[0].leadfield
            L_s = leadfield @ source_covariance
            L = leadfield
            W = np.diag(np.linalg.norm(L, axis=0))
            inverse_operator = (
                source_covariance @ np.linalg.inv(L_s.T @ L_s + W.T @ W) @ L_s.T
            )
            stc.data = inverse_operator @ data
        elif self.summary_type.lower() == "normalized_mean":
            # Normalize the data
            data = np.stack(
                [stc_.data / np.linalg.norm(stc_.data) for stc_ in stc_list], axis=0
            )
            # Calculate the mean of the normalized data
            stc.data = np.nanmean(data, axis=0)

        else:
            msg = f"summary_type is {self.summary_type} but must be either of the following: {self.summary_types}"
            raise AttributeError(msg)
        return stc

    def calc_neg_log_likelihood(self, y, x):
        L = self.solvers[0].leadfield
        Gamma = np.diag(np.mean(abs(y), axis=-1))
        C_y = (x @ x.T) / x.shape[-1]
        sigma_y = L @ Gamma @ L.T
        sigma_y_inv = np.linalg.inv(sigma_y)
        _, logdet_sigma_y_estimated = np.linalg.slogdet(sigma_y)
        neg_log_likelihood = logdet_sigma_y_estimated + abs(np.trace(C_y + sigma_y_inv))
        return neg_log_likelihood
