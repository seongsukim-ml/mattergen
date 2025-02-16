from typing import Callable
from typing import Generic, Mapping, Tuple, TypeVar

from tqdm.auto import tqdm

from mattergen.diffusion.corruption.multi_corruption import MultiCorruption, apply
from mattergen.diffusion.data.batched_data import BatchedData
from mattergen.diffusion.diffusion_module import DiffusionModule
from mattergen.diffusion.lightning_module import DiffusionLightningModule
from mattergen.diffusion.sampling.pc_partials import CorrectorPartial, PredictorPartial
from mattergen.self_guidance.symmetry_utils import *


Diffusable = TypeVar(
    "Diffusable", bound=BatchedData
)  # Don't use 'T' because it clashes with the 'T' for time
SampleAndMean = Tuple[Diffusable, Diffusable]
SampleAndMeanAndMaybeRecords = Tuple[Diffusable, Diffusable, list[Diffusable] | None]
SampleAndMeanAndRecords = Tuple[Diffusable, Diffusable, list[Diffusable]]

import torch

from mattergen.diffusion.sampling.pc_sampler import Diffusable
from mattergen.self_guidance.wyckoff_pc_sampler import PredictorCorrector

from mattergen.diffusion.corruption.multi_corruption import apply
from mattergen.diffusion.model_utils import convert_model_out_to_score
from mattergen.self_guidance.symmetry_utils import *
from mattergen.common.utils.data_utils import compute_lattice_polar_decomposition
from mattergen.diffusion.sampling.pc_sampler import _mask_replace

from torch_scatter import scatter


def d_log_p_wrapped_normal(x, sigma, N=10, T=1.0):
    """Calculate the gradient of the log wrapped normal probability density."""
    if sigma.dim() == x.dim():
        sigma = sigma.unsqueeze(-1)
    indices = torch.arange(-N, N + 1, device=x.device, dtype=x.dtype)  # Vectorized indices
    shifts = T * indices  # Periodic shifts
    x_shifted = x.unsqueeze(-1) + shifts  # Broadcast shifts across x

    exp_terms = torch.exp(-(x_shifted**2) / (2 * sigma**2))  # Compute exponential term
    numerator = (x_shifted / sigma**2) * exp_terms  # Gradient contribution
    denominator = exp_terms.sum(dim=-1)  # Sum of probabilities (normalizing constant)

    grad = numerator.sum(dim=-1) / denominator  # Compute log-derivative
    return grad


BatchTransform = Callable[[Diffusable], Diffusable]


def identity(x: Diffusable) -> Diffusable:
    """
    Default function that transforms data to its conditional state
    """
    return x


class GuidedPredictorCorrector(PredictorCorrector):
    """
    Sampler for classifier-free guidance.
    """

    def __init__(
        self,
        *,
        guidance_scale: float,
        remove_conditioning_fn: BatchTransform | None = None,
        keep_conditioning_fn: BatchTransform | None = None,
        guidance_start: float | None = None,
        guidance_end: float | None = None,
        use_cond_model: False,
        ts: float = 1.0,
        te: float = 0.0,
        **kwargs,
    ):
        """
        guidance_scale: gamma in p_gamma(x|y)=p(x)p(y|x)**gamma for classifier-free guidance
        remove_conditioning_fn: function that removes conditioning from the data
        keep_conditioning_fn: function that will be applied to the data before evaluating the conditional score. For example, this function might drop some fields that you never want to condition on or add fields that indicate which conditions should be respected.
        **kwargs: passed on to parent class constructor.
        """

        super().__init__(**kwargs)
        self._remove_conditioning_fn = remove_conditioning_fn
        self._keep_conditioning_fn = keep_conditioning_fn or identity
        self._guidance_scale = guidance_scale

        self._guidance_start = guidance_start
        self._guidance_end = guidance_end
        self.cf = CrystalFamily()
        self.lat_perm, self.perm_for_A1, self.perm_for_A2 = get_latttice_permutations(
            device=self._device
        )
        self.anchors = None
        self.uniques = None
        self.wyckoff_batch = None
        self.frac_batch = None
        self.step = 0
        self.prev_pos_target = None
        self.prev_cell_target = None
        self.ts = ts
        self.te = te

    def _score_fn(
        self,
        x: Diffusable,
        t: torch.Tensor,
    ) -> Diffusable:
        """For each field, regardless of whether the corruption process is SDE or D3PM, we guide the score in the same way here,
        by taking a linear combination of the conditional and unconditional score model output.

        For discrete fields, the score model outputs are interpreted as logits, so the linear combination here means we compute logits for
        p_\gamma(x|y)=p(x)^(1-\gamma) p(x|y)^\gamma

        """

        def get_unconditional_score():
            return super(GuidedPredictorCorrector, self)._score_fn(
                x=self._remove_conditioning_fn(x), t=t
            )

        def get_conditional_score():
            return super(GuidedPredictorCorrector, self)._score_fn(
                x=self._keep_conditioning_fn(x), t=t
            )

        if abs(self._guidance_scale - 1) < 1e-15:
            return get_conditional_score()
        elif abs(self._guidance_scale) < 1e-15:
            return get_unconditional_score()
        else:
            # guided_score = guidance_factor * conditional_score + (1-guidance_factor) * unconditional_score

            conditional_score = get_conditional_score()
            unconditional_score = get_unconditional_score()
            return unconditional_score.replace(
                **{
                    k: torch.lerp(
                        unconditional_score[k], conditional_score[k], self._guidance_scale
                    )
                    for k in self._multi_corruption.corrupted_fields
                }
            )

    def _score_fn_guidance(
        self,
        x: Diffusable,
        t: torch.Tensor,
    ) -> Diffusable:
        """For each field, regardless of whether the corruption process is SDE or D3PM, we guide the score in the same way here,
        by taking a linear combination of the conditional and unconditional score model output.

        For discrete fields, the score model outputs are interpreted as logits, so the linear combination here means we compute logits for
        p_\gamma(x|y)=p(x)^(1-\gamma) p(x|y)^\gamma

        """
        # Initialize anchors and uniques
        if (
            self.frac_batch is None
            or self.frac_batch.shape[0] != x.wyckoff_bat.shape[0]
            or self.step == 0
        ):
            self.anchors = None
            self.uniques = None
            self.wyckoff_batch = None
            self.frac_batch = None
            self.prev_pos_target = None
            self.prev_cell_target = None

        if self.anchors is None:
            anchors = x.anchors.clone().detach()
            idx = 0
            cum = 0
            for len, num_atom in zip(x.anchors_len, x.num_atoms):
                anchors[idx : idx + len] = anchors[idx : idx + len] + cum
                idx += len
                cum += num_atom
            self.anchors = anchors
        else:
            anchors = self.anchors
        assert anchors.shape[0] == x.pos.shape[0]

        if self.uniques is None:
            uniques = x.uniques.clone().detach()
            idx = 0
            for len in x.uniques_len:
                uniques[idx : idx + len] = uniques[idx : idx + len] + idx
                idx += len
            self.uniques = uniques
        else:
            uniques = self.uniques

        if self.wyckoff_batch is None:
            wyckoff_batch = x.wyckoff_bat.clone().detach()
            idx = 0
            cum = 0
            for len, num_atom in zip(x.wyckoff_bat_len, x.num_atoms):
                wyckoff_batch[idx : idx + len] = wyckoff_batch[idx : idx + len] + cum
                idx += len
                cum += num_atom
            self.wyckoff_batch = wyckoff_batch  # It double the value of the wyckoff_batch since x.wyckoff_batch already has the batch sum
        else:
            wyckoff_batch = self.wyckoff_batch

        def get_unconditional_score():
            x_no_conditioning = self._remove_conditioning_fn(x)
            pred_data = self.diffusion_module.model(x_no_conditioning, t)

            fns = {
                k: convert_model_out_to_score for k in self.diffusion_module.corruption.sdes.keys()
            }

            scores = apply(
                fns=fns,
                model_out=pred_data,
                broadcast=dict(t=t, batch=x_no_conditioning),
                sde=self.diffusion_module.corruption.sdes,
                model_target=self.diffusion_module.model_targets,
                batch_idx=self.diffusion_module.corruption._get_batch_indices(x),
            )
            if (self.step + 1) in [
                1,
                100,
                250,
                500,
                750,
                1000,
            ] and self.mode == "predict":
                print(f"Step {self.step}")
                print("~~~~~~~~~~~~~~~~~~~~~~")
                # print(f"pos score: {pos_score2}")
                print(f"pos_pred: {pred_data.pos}")
                print("~~~~~~~~~~~~~~~~~~~~~~")
                # print(f"lat score: {lattice_score2}")
                print(f"lat_pred: {pred_data.cell}")
                print("~~~~~~~~~~~~~~~~~~~~~~")
                # print(f"pos_target: {pos_target}")
                print(f"pos: {x.pos}")
                print("~~~~~~~~~~~~~~~~~~~~~~")
                # print(f"cell_target: {cell_target}")
                print(f"cell: {x.cell}")

            return pred_data.clone().replace(**scores), pred_data

        def get_conditional_score():
            with torch.set_grad_enabled(True):
                x.cell.requires_grad = True
                x.pos.requires_grad = True
                x_no_conditioning = self._remove_conditioning_fn(x)
                pred_data = self.diffusion_module.model(x_no_conditioning, t)

                fns = {
                    k: convert_model_out_to_score
                    for k in self.diffusion_module.corruption.sdes.keys()
                }
                scores = apply(
                    fns=fns,
                    model_out=pred_data,
                    broadcast=dict(t=t, batch=x_no_conditioning),
                    sde=self.diffusion_module.corruption.sdes,
                    model_target=self.diffusion_module.model_targets,
                    batch_idx=self.diffusion_module.corruption._get_batch_indices(x),
                )

                c_mean, c_std = self.diffusion_module.corruption.sdes["cell"].mean_coeff_and_std(
                    x.cell,
                    t,
                    batch=x,
                    batch_idx=self.diffusion_module.corruption._get_batch_indices(x)["cell"],
                )
                c_lim_mean = self.diffusion_module.corruption.sdes["cell"].get_limit_mean(
                    x=x.cell, batch=x
                )
                p_mean, p_std = self.diffusion_module.corruption.sdes["pos"].mean_coeff_and_std(
                    x.pos,
                    t,
                    batch=x,
                    batch_idx=self.diffusion_module.corruption._get_batch_indices(x)["pos"],
                )

                pred_data = pred_data.replace(**scores)
                no_cond = pred_data.clone().detach()

                proj_pos, proj_cell = x.pos, x.cell
                proj_pos = proj_pos.detach()
                proj_cell = proj_cell.detach()

                # pos_target = proj_pos
                # pos_target2 = x.proj_pos
                pos_target = x.proj_pos
                pos_target2 = proj_pos

                clean_pos_diff = (x.pos - pos_target - torch.round(x.pos - pos_target)).pow(2)
                # clean_pos_diff2 = (x.pos - pos_target2 - torch.round(x.pos - pos_target2)).pow(2)
                # pos_mask = torch.isclose(pos_target, pos_target2).float().detach()
                # clean_pos_diff = clean_pos_diff * (1 + pos_mask * 100 * (1 - t[0]))
                # clean_pos_diff = clean_pos_diff * (1 + pos_mask * 10)
                # if pos_mask.sum() == 0:
                #     clean_pos_diff = clean_pos_diff * 10

                pos_score = torch.autograd.grad(
                    clean_pos_diff.sum(),
                    x.pos,
                    allow_unused=True,
                    create_graph=True,
                )[0].detach()
                pos_coeff = 0.03 / p_std / p_std
                # pos_coeff = 1
                pos_score2 = -pos_score * pos_coeff
                scores["pos"] = pos_score2
                # scores["pos"] = torch.zeros_like(pos_score2)
                x.pos.detach()

                # cell_target = compute_lattice_polar_decomposition(x.proj_cell)
                # cell_target = x.proj_cell
                # cell_target2 = compute_lattice_polar_decomposition(proj_cell)
                cell_target = x.proj_cell
                cell_target2 = proj_cell

                clean_lattice_diff = (cell_target - x.cell).pow(2)
                # clean_lattice_diff2 = (cell_target2 - x.cell).pow(2)
                cell_mask = torch.isclose(cell_target, cell_target2).float().detach()
                # clean_lattice_diff = clean_lattice_diff + clean_lattice_diff2
                # clean_lattice_diff = clean_lattice_diff * (1 + cell_mask * 100 * (1 - t[0]))
                # clean_lattice_diff = clean_lattice_diff * (1 + cell_mask * 10)
                clean_lattice_diff = clean_lattice_diff * (1 + cell_mask * 10)
                # if cell_mask.sum() == 0:
                #     clean_lattice_diff = clean_lattice_diff * 10

                lattice_score = torch.autograd.grad(
                    clean_lattice_diff.sum(),
                    x.cell,
                    allow_unused=True,
                )[0].detach()
                lattice_coeff = 1 / c_std
                # lattice_coeff = 1
                lattice_score2 = -lattice_score * lattice_coeff
                scores["cell"] = lattice_score2
                # scores["cell"] = torch.zeros_like(lattice_score2)
                x.cell.detach()

                if (self.step + 1) in [
                    1,
                    100,
                    250,
                    500,
                    750,
                    1000,
                ] and self.mode == "predict":
                    print(f"\nStep {self.step+1}")
                    print(f"pos score: {pos_score2}")
                    print(
                        f"pos score mean, max: {pos_score2.abs().mean()}, {pos_score2.abs().max()}"
                    )
                    print(f"pos_pred: {pred_data.pos}")
                    print(
                        f"pos_pred mean, max: {pred_data.pos.abs().mean()}, {pred_data.pos.abs().max()}"
                    )
                    print(
                        f"pos score ratio mean, max: {pos_score2.abs().mean()/pred_data.pos.abs().mean()}, {pos_score2.abs().max()/pred_data.pos.abs().max()}"
                    )
                    print(f"Step {self.step+1}~~~~~~~~~~~~~~~~~~~~~~")
                    print(f"lat score: {lattice_score2}")
                    print(
                        f"lat score mean, max: {lattice_score2.abs().mean()}, {lattice_score2.abs().max()}"
                    )
                    print(f"lat_pred: {pred_data.cell}")
                    print(
                        f"lat_pred mean, max: {pred_data.cell.abs().mean()}, {pred_data.cell.abs().max()}"
                    )
                    print(
                        f"lat score ratio mean, max: {lattice_score2.abs().mean()/pred_data.cell.abs().mean()}, {lattice_score2.abs().max()/pred_data.cell.abs().max()}"
                    )
                    print(f"Step {self.step+1}~~~~~~~~~~~~~~~~~~~~~~")
                    print(f"pos_target: {pos_target}")
                    print(f"pos: {x.pos}")
                    # print(f"pos_mask: {pos_mask}")
                    print(f"Step {self.step+1}~~~~~~~~~~~~~~~~~~~~~~")
                    print(f"cell_target: {cell_target}")
                    print(f"cell: {x.cell}")
                    print(f"cell_mask: {cell_mask}")
                    print(f"Step {self.step+1} guidance_scale: {_guidance_scale}")
                    print(f"c_std: {c_std[0][0][0]}, p_std: {p_std[0][0]}")
                    print(
                        f"pos score ratio mean, max: {pos_score2.abs().mean()/pred_data.pos.mean()}, {pos_score2.abs().max()/pred_data.pos.abs().max()}"
                    )
                    print(
                        f"lat score ratio mean, max: {lattice_score2.abs().mean()/pred_data.cell.mean()}, {lattice_score2.abs().max()/pred_data.cell.abs().max()}"
                    )
            # scores["atomic_numbers"] = pred_data.atomic_numbers[anchors]
            return no_cond, pred_data.replace(**scores)

        _guidance_scale = 0
        # ts = 0.95
        ts = self.ts
        te = self.te
        if t[0] <= ts and t[0] >= te:
            if self._guidance_start is not None:
                _guidance_scale = self._guidance_start + (
                    self._guidance_end - self._guidance_start
                ) * (ts - t[0])
            else:
                _guidance_scale = self._guidance_scale

        if abs(_guidance_scale) < 1e-15:
            return get_unconditional_score()[0]
        else:
            # guided_score = guidance_factor * conditional_score + (1-guidance_factor) * unconditional_score

            # unconditional_score, model_out = get_unconditional_score()
            unconditional_score, conditional_score = get_conditional_score()
            ret = unconditional_score.replace(
                **{
                    k: (unconditional_score[k] + _guidance_scale * conditional_score[k])
                    for k in self._multi_corruption.corrupted_fields
                }
            )
            return ret

    @torch.no_grad()
    def _denoise(
        self,
        batch: Diffusable,
        mask: dict[str, torch.Tensor],
        record: bool = False,
    ) -> SampleAndMeanAndMaybeRecords:
        """Denoise from a prior sample to a t=eps_t sample."""
        recorded_samples = None
        if record:
            recorded_samples = []
        for k in self._predictors:
            mask.setdefault(k, None)
        for k in self._correctors:
            mask.setdefault(k, None)
        mean_batch = batch.clone()
        proj_batch = batch.clone()
        proj_mean_batch = batch.clone()
        proj_pos_noise, proj_cell_noise = self._project_to_space_group(batch)
        proj_batch = proj_batch.replace(pos=proj_pos_noise, cell=proj_cell_noise)
        proj_mean_batch = proj_mean_batch.replace(pos=proj_pos_noise, cell=proj_cell_noise)

        # Decreasing timesteps from T to eps_t
        timesteps = torch.linspace(self._max_t, self._eps_t, self.N, device=self._device)
        dt = -torch.tensor((self._max_t - self._eps_t) / (self.N - 1)).to(self._device)

        for i in tqdm(range(self.N), miniters=50, mininterval=5):
            self.step = i
            # Set the timestep
            t = torch.full((batch.get_batch_size(),), timesteps[i], device=self._device)

            # Corrector updates.
            if self._correctors:
                for _ in range(self._n_steps_corrector):
                    self.mode = "correct"
                    score = self._score_fn(batch, t)
                    fns = {
                        k: corrector.step_given_score for k, corrector in self._correctors.items()
                    }
                    samples_means: dict[str, Tuple[torch.Tensor, torch.Tensor]] = apply(
                        fns=fns,
                        broadcast={"t": t},
                        x=batch,
                        score=score,
                        batch_idx=self._multi_corruption._get_batch_indices(batch),
                    )
                    if record:
                        recorded_samples.append(batch.clone().to("cpu"))
                    batch, mean_batch = _mask_replace(
                        samples_means=samples_means, batch=batch, mean_batch=mean_batch, mask=mask
                    )

                    proj_pos, proj_cell = self._project_to_space_group(batch)
                    proj_batch = proj_batch.replace(proj_pos=proj_pos, proj_cell=proj_cell)
                    proj_mean_batch = proj_mean_batch.replace(
                        proj_pos=proj_pos, proj_cell=proj_cell
                    )

                    score_proj = self._score_fn_guidance(proj_batch, t)
                    fns = {
                        k: corrector.step_given_score for k, corrector in self._correctors.items()
                    }
                    samples_proj_means: dict[str, Tuple[torch.Tensor, torch.Tensor]] = apply(
                        fns=fns,
                        broadcast={"t": t},
                        x=proj_batch,
                        score=score_proj,
                        batch_idx=self._multi_corruption._get_batch_indices(proj_batch),
                    )
                    proj_batch, proj_mean_batch = _mask_replace(
                        samples_means=samples_proj_means,
                        batch=proj_batch,
                        mean_batch=proj_mean_batch,
                        mask=mask,
                    )
                    new_proj_pos, new_proj_cell = self._project_to_space_group(proj_batch)
                    proj_batch = proj_batch.replace(pos=new_proj_pos, cell=new_proj_cell)
                    proj_mean_batch = proj_mean_batch.replace(pos=new_proj_pos, cell=new_proj_cell)

            # Predictor updates
            self.mode = "predict"
            score = self._score_fn(batch, t)
            predictor_fns = {
                k: predictor.update_given_score for k, predictor in self._predictors.items()
            }
            samples_means = apply(
                fns=predictor_fns,
                x=batch,
                score=score,
                broadcast=dict(t=t, batch=batch, dt=dt),
                batch_idx=self._multi_corruption._get_batch_indices(batch),
            )
            if record:
                recorded_samples.append(batch.clone().to("cpu"))
            batch, mean_batch = _mask_replace(
                samples_means=samples_means, batch=batch, mean_batch=mean_batch, mask=mask
            )

            proj_pos, proj_cell = self._project_to_space_group(batch)
            proj_batch = proj_batch.replace(proj_pos=proj_pos, proj_cell=proj_cell)
            proj_mean_batch = proj_mean_batch.replace(proj_pos=proj_pos, proj_cell=proj_cell)

            score_proj = self._score_fn_guidance(proj_batch, t)
            predictor_fns = {
                k: predictor.update_given_score for k, predictor in self._predictors.items()
            }
            samples_proj_means = apply(
                fns=predictor_fns,
                x=proj_batch,
                score=score_proj,
                broadcast=dict(t=t, batch=proj_batch, dt=dt),
                batch_idx=self._multi_corruption._get_batch_indices(proj_batch),
            )
            proj_batch, proj_mean_batch = _mask_replace(
                samples_means=samples_proj_means,
                batch=proj_batch,
                mean_batch=proj_mean_batch,
                mask=mask,
            )
            new_proj_pos, new_proj_cell = self._project_to_space_group(proj_batch)
            proj_batch = proj_batch.replace(pos=new_proj_pos, cell=new_proj_cell)
            proj_mean_batch = proj_mean_batch.replace(pos=new_proj_pos, cell=new_proj_cell)
            self._project_to_space_group(proj_batch.replace(pos=new_proj_pos, cell=new_proj_cell))[
                1
            ]
            # proj_batch = proj_batch.replace(atomic_numbers=sym_atom)
            # proj_mean_batch = proj_mean_batch.replace(atomic_numbers=sym_atom)
        sym_atom_freq = scatter(
            torch.functional.F.one_hot(proj_batch.atomic_numbers),
            self.anchors,
            dim=0,
            reduce="sum",
        )
        sym_atom = torch.argmax(sym_atom_freq, dim=-1)[self.anchors]
        proj_batch = proj_batch.replace(atomic_numbers=sym_atom)
        proj_mean_batch = proj_mean_batch.replace(atomic_numbers=sym_atom)

        final_proj_batch = proj_batch.clone()
        fin_proj_pos, fin_proj_cell = self._project_to_space_group(final_proj_batch)
        final_proj_batch = final_proj_batch.replace(
            pos=fin_proj_pos,
            cell=fin_proj_cell,
            atomic_numbers=final_proj_batch.atomic_numbers[self.anchors],
        )

        return (
            proj_batch,
            proj_mean_batch,
            recorded_samples,
            final_proj_batch,
            batch.clone().replace(pos=proj_pos % 1.0, cell=proj_cell),
        )

    def _project_to_space_group(self, batch):
        cf = self.cf
        cf.set_device(batch.pos.device)
        lat_perm, _, perm_for_A2 = get_latttice_permutations2(device=batch.pos.device)

        wyckoff_batch = batch.wyckoff_bat.clone()
        idx, cum = 0, 0
        for len, num_atom in zip(batch.wyckoff_bat_len, batch.num_atoms):
            wyckoff_batch[idx : idx + len] = wyckoff_batch[idx : idx + len] + cum
            idx += len
            cum += num_atom

        # project latice
        conv_lat = torch.bmm(batch.prim_to_conv, batch.cell)
        conv_lat_vec = cf.m2v(cf.de_so3(conv_lat))
        conv_lat_vec_proj = cf.proj_k_to_spacegroup(conv_lat_vec, batch.space_groups)
        conv_lat_proj = cf.v2m(conv_lat_vec_proj)

        rank = torch.argsort(-torch.norm(conv_lat_proj, dim=-1), dim=-1)
        idx = torch.cat([batch.space_groups.unsqueeze(-1), rank], dim=-1)
        perm = lat_perm[idx[:, 0], idx[:, 1], idx[:, 2], idx[:, 3]]

        perm_conv_lat_proj = torch.bmm(torch.bmm(perm, conv_lat_proj), perm.transpose(-1, -2))

        rankA = torch.argsort(-torch.norm(conv_lat_proj, dim=-1), dim=-1)
        idxA = torch.cat([batch.space_groups.unsqueeze(-1), rankA], dim=-1)
        permA = perm_for_A2[idxA[:, 0], idxA[:, 1], idxA[:, 2], idxA[:, 3]]

        prim_lat_proj = torch.bmm(
            batch.conv_to_prim,
            torch.bmm(torch.bmm(permA, perm_conv_lat_proj), permA.transpose(-1, -2)),
        )
        pos_cart = torch.einsum("bi,bij->bj", batch.pos, batch.cell[batch.batch])
        pos_cart = torch.einsum("bij,bj->bi", perm[batch.batch], pos_cart)
        pos_frac_conv = torch.einsum(
            "bi,bij->bj", pos_cart, torch.inverse(perm_conv_lat_proj)[batch.batch]
        )
        pos_frac_conv = pos_frac_conv % 1.0
        pos_tran = torch.cat(
            [
                pos_frac_conv[wyckoff_batch],
                torch.ones(pos_frac_conv[wyckoff_batch].shape[0], 1, device=batch.pos.device),
            ],
            dim=1,
        )

        pos_frac_proj = (
            torch.einsum("bij,bj->bi", batch.wyckoff_ops, pos_tran).squeeze(-1)[:, :3] % 1.0
        )
        pos_cart_proj = torch.einsum(
            "bi,bij->bj", pos_frac_proj, perm_conv_lat_proj[batch.batch[wyckoff_batch]]
        )

        prim_lat_proj = prim_lat_proj

        prim_lat_inv = torch.inverse(prim_lat_proj)
        pos_cart_proj = torch.einsum("bij,bj->bi", permA[batch.batch[wyckoff_batch]], pos_cart_proj)
        pos_prim_frac_proj_all = torch.einsum(
            "bi,bij->bj", pos_cart_proj, prim_lat_inv[batch.batch[wyckoff_batch]]
        )

        ## Get prim idx
        if self.frac_batch is None or self.frac_batch.shape[0] != wyckoff_batch.shape[0]:
            for i in range(10):
                random_pos_frac_conv = torch.rand_like(pos_frac_conv).to(pos_frac_conv.device)
                random_pos_tran = torch.cat(
                    [
                        random_pos_frac_conv[wyckoff_batch],
                        torch.ones(
                            random_pos_frac_conv[wyckoff_batch].shape[0],
                            1,
                            device=pos_frac_conv.device,
                        ),
                    ],
                    dim=1,
                )
                random_pos_frac_proj = (
                    torch.einsum("bij,bj->bi", batch.wyckoff_ops, random_pos_tran).squeeze(-1)[
                        :, :3
                    ]
                    % 1.0
                ) % 1.0
                random_pos_cart_proj = torch.einsum(
                    "bi,bij->bj",
                    random_pos_frac_proj,
                    perm_conv_lat_proj[batch.batch[wyckoff_batch]],
                )
                random_pos_cart_proj = torch.einsum(
                    "bij,bj->bi", permA[batch.batch[wyckoff_batch]], random_pos_cart_proj
                )

                random_fracs = torch.einsum(
                    "bi,bij->bj",
                    random_pos_cart_proj,
                    prim_lat_inv[batch.batch[wyckoff_batch]],
                )
                random_fracs = random_fracs % 1.0
                random_fracs_diff = random_fracs.unsqueeze(1) - random_fracs.unsqueeze(0)
                random_fracs_diff = random_fracs_diff - torch.round(random_fracs_diff)
                EPSILON = 5e-4
                random_fracs_diff_is_zero = torch.all(
                    torch.isclose(
                        random_fracs_diff,
                        torch.zeros_like(random_fracs_diff),
                        rtol=EPSILON,
                        atol=EPSILON,
                    ),
                    dim=-1,
                )
                random_fracs_idx = random_fracs_diff_is_zero & (
                    wyckoff_batch.unsqueeze(0) == wyckoff_batch.unsqueeze(1)
                )
                random_fracs_idx = ~(random_fracs_idx.triu(diagonal=1).any(dim=0))
                # random_fracs_prim = random_fracs[random_fracs_idx]
                assert random_fracs_idx.shape[0] == pos_prim_frac_proj_all.shape[0]
                pos_prim_frac_proj = pos_prim_frac_proj_all[random_fracs_idx]
                if pos_prim_frac_proj.shape[0] == batch.pos.shape[0]:
                    self.frac_batch = random_fracs_idx
                    break
        pos_prim_frac_proj = pos_prim_frac_proj_all[self.frac_batch]

        return (pos_prim_frac_proj % 1.0), prim_lat_proj

    # def _project_to_space_group(self, batch):
    #     cf = self.cf
    #     cf.set_device(batch.pos.device)
    #     lat_perm, _, perm_for_A2 = get_latttice_permutations(device=batch.pos.device)

    #     wyckoff_batch = batch.wyckoff_bat.clone()
    #     idx, cum = 0, 0
    #     for len, num_atom in zip(batch.wyckoff_bat_len, batch.num_atoms):
    #         wyckoff_batch[idx : idx + len] = wyckoff_batch[idx : idx + len] + cum
    #         idx += len
    #         cum += num_atom

    #     # project latice
    #     conv_lat = torch.bmm(batch.prim_to_conv, batch.cell)
    #     conv_lat_vec = cf.m2v(cf.de_so3(conv_lat))
    #     conv_lat_vec_proj = cf.proj_k_to_spacegroup(conv_lat_vec, batch.space_groups)
    #     conv_lat_proj = cf.v2m(conv_lat_vec_proj)

    #     rank = torch.argsort(-torch.norm(conv_lat_proj, dim=-1), dim=-1)
    #     idx = torch.cat([batch.space_groups.unsqueeze(-1), rank], dim=-1)
    #     perm = lat_perm[idx[:, 0], idx[:, 1], idx[:, 2], idx[:, 3]]
    #     perm_A = perm_for_A2[batch.space_groups]

    #     perm_conv_lat_proj = torch.bmm(torch.bmm(perm, conv_lat_proj), perm.transpose(-1, -2))
    #     perm_conv_lat_proj = torch.bmm(
    #         torch.bmm(perm_A, perm_conv_lat_proj), perm_A.transpose(-1, -2)
    #     )

    #     prim_lat_proj = torch.bmm(batch.conv_to_prim, perm_conv_lat_proj)

    #     pos_cart = torch.einsum("bi,bij->bj", batch.pos, batch.cell[batch.batch])
    #     pos_frac_conv = torch.einsum(
    #         "bi,bij->bj", pos_cart, torch.inverse(perm_conv_lat_proj)[batch.batch]
    #     )
    #     pos_tran = torch.cat(
    #         [
    #             pos_frac_conv[wyckoff_batch],
    #             torch.ones(pos_frac_conv[wyckoff_batch].shape[0], 1, device=batch.pos.device),
    #         ],
    #         dim=1,
    #     )

    #     pos_frac_proj = (
    #         torch.einsum("bij,bj->bi", batch.wyckoff_ops, pos_tran).squeeze(-1)[:, :3] % 1.0
    #     )
    #     pos_frac_proj = torch.einsum("bij,bj->bi", perm[batch.batch[wyckoff_batch]], pos_frac_proj)
    #     pos_cart_porj = torch.einsum(
    #         "bi,bij->bj", pos_frac_proj, perm_conv_lat_proj[batch.batch[wyckoff_batch]]
    #     )

    #     prim_lat_inv = torch.inverse(prim_lat_proj)
    #     pos_prim_frac_proj_all = torch.einsum(
    #         "bi,bij->bj", pos_cart_porj, prim_lat_inv[batch.batch[wyckoff_batch]]
    #     )

    #     ## Get prim idx
    #     if self.frac_batch is None or self.frac_batch.shape[0] != wyckoff_batch.shape[0]:
    #         for i in range(10):
    #             random_pos_frac_conv = torch.rand_like(pos_frac_conv).to(pos_frac_conv.device)
    #             random_pos_tran = torch.cat(
    #                 [
    #                     random_pos_frac_conv[wyckoff_batch],
    #                     torch.ones(
    #                         random_pos_frac_conv[wyckoff_batch].shape[0],
    #                         1,
    #                         device=pos_frac_conv.device,
    #                     ),
    #                 ],
    #                 dim=1,
    #             )
    #             random_pos_frac_proj = (
    #                 torch.einsum("bij,bj->bi", batch.wyckoff_ops, random_pos_tran).squeeze(-1)[
    #                     :, :3
    #                 ]
    #                 % 1.0
    #             ) % 1.0
    #             random_pos_frac_proj = torch.einsum(
    #                 "bij,bj->bi", perm[batch.batch[wyckoff_batch]], random_pos_frac_proj
    #             )
    #             random_pos_cart_proj = torch.einsum(
    #                 "bi,bij->bj",
    #                 random_pos_frac_proj,
    #                 perm_conv_lat_proj[batch.batch[wyckoff_batch]],
    #             )

    #             random_fracs = torch.einsum(
    #                 "bi,bij->bj",
    #                 random_pos_cart_proj,
    #                 prim_lat_inv[batch.batch[wyckoff_batch]],
    #             )
    #             random_fracs = random_fracs % 1.0
    #             random_fracs_diff = random_fracs.unsqueeze(1) - random_fracs.unsqueeze(0)
    #             random_fracs_diff = random_fracs_diff - torch.round(random_fracs_diff)
    #             EPSILON = 5e-4
    #             random_fracs_diff_is_zero = torch.all(
    #                 torch.isclose(
    #                     random_fracs_diff,
    #                     torch.zeros_like(random_fracs_diff),
    #                     rtol=EPSILON,
    #                     atol=EPSILON,
    #                 ),
    #                 dim=-1,
    #             )
    #             random_fracs_idx = random_fracs_diff_is_zero & (
    #                 wyckoff_batch.unsqueeze(0) == wyckoff_batch.unsqueeze(1)
    #             )
    #             random_fracs_idx = ~(random_fracs_idx.triu(diagonal=1).any(dim=0))
    #             # random_fracs_prim = random_fracs[random_fracs_idx]
    #             assert random_fracs_idx.shape[0] == pos_prim_frac_proj_all.shape[0]
    #             pos_prim_frac_proj = pos_prim_frac_proj_all[random_fracs_idx]
    #             if pos_prim_frac_proj.shape[0] == batch.pos.shape[0]:
    #                 self.frac_batch = random_fracs_idx
    #                 break
    #     pos_prim_frac_proj = pos_prim_frac_proj_all[self.frac_batch]
    #     # prim_lat_proj = torch.bmm(torch.bmm(perm_A, prim_lat_proj2),perm_A.transpose(-1,-2))
    #     # pos_prim_frac_proj = torch.einsum("bij,bj->bi", perm_A, pos_prim_frac_proj)

    #     return (pos_prim_frac_proj % 1.0), prim_lat_proj

    # def _project_to_space_group_epsilon(self, batch):
    #     cf = self.cf
    #     cf.set_device(batch.pos.device)
    #     lat_perm, _, perm_for_A2 = get_latttice_permutations(device=batch.pos.device)

    #     wyckoff_batch = batch.wyckoff_bat.clone()
    #     idx, cum = 0, 0
    #     for len, num_atom in zip(batch.wyckoff_bat_len, batch.num_atoms):
    #         wyckoff_batch[idx : idx + len] = wyckoff_batch[idx : idx + len] + cum
    #         idx += len
    #         cum += num_atom

    #     # project latice
    #     conv_lat = torch.bmm(batch.prim_to_conv, batch.cell)
    #     conv_lat_vec = cf.m2v(cf.de_so3(conv_lat))
    #     conv_lat_vec_proj = cf.proj_k_to_spacegroup(conv_lat_vec, batch.space_groups)
    #     conv_lat_proj = cf.v2m(conv_lat_vec_proj)

    #     rank = torch.argsort(-torch.norm(conv_lat_proj, dim=-1), dim=-1)
    #     idx = torch.cat([batch.space_groups.unsqueeze(-1), rank], dim=-1)
    #     perm = lat_perm[idx[:, 0], idx[:, 1], idx[:, 2], idx[:, 3]]
    #     perm_A = perm_for_A2[batch.space_groups]

    #     perm_conv_lat_proj = torch.bmm(torch.bmm(perm, conv_lat_proj), perm.transpose(-1, -2))
    #     perm_conv_lat_proj = torch.bmm(
    #         torch.bmm(perm_A, perm_conv_lat_proj), perm_A.transpose(-1, -2)
    #     )

    #     prim_lat_proj = torch.bmm(batch.conv_to_prim, perm_conv_lat_proj)

    #     pos_cart = torch.einsum("bi,bij->bj", batch.pos, batch.cell[batch.batch])
    #     pos_frac_conv = torch.einsum(
    #         "bi,bij->bj", pos_cart, torch.inverse(perm_conv_lat_proj)[batch.batch]
    #     )
    #     pos_tran = pos_frac_conv[wyckoff_batch]
    #     # pos_tran = torch.cat(
    #     #     [
    #     #         pos_frac_conv[wyckoff_batch],
    #     #         torch.ones(pos_frac_conv[wyckoff_batch].shape[0], 1, device=batch.pos.device),
    #     #     ],
    #     #     dim=1,
    #     # )

    #     # pos_frac_proj = (
    #     #     torch.einsum("bij,bj->bi", batch.wyckoff_ops, pos_tran).squeeze(-1)[:, :3] % 1.0
    #     # )
    #     pos_frac_proj = (
    #         torch.einsum("bij,bj->bi", batch.wyckoff_ops[:, :3, :3], pos_tran).squeeze(-1)[:, :3]
    #         % 1.0
    #     )
    #     pos_frac_proj = torch.einsum("bij,bj->bi", perm[batch.batch[wyckoff_batch]], pos_frac_proj)
    #     pos_cart_porj = torch.einsum(
    #         "bi,bij->bj", pos_frac_proj, perm_conv_lat_proj[batch.batch[wyckoff_batch]]
    #     )

    #     prim_lat_inv = torch.inverse(prim_lat_proj)
    #     pos_prim_frac_proj_all = torch.einsum(
    #         "bi,bij->bj", pos_cart_porj, prim_lat_inv[batch.batch[wyckoff_batch]]
    #     )

    #     ## Get prim idx
    #     if self.frac_batch is None or self.frac_batch.shape[0] != wyckoff_batch.shape[0]:
    #         for i in range(10):
    #             random_pos_frac_conv = torch.rand_like(pos_frac_conv).to(pos_frac_conv.device)
    #             random_pos_tran = torch.cat(
    #                 [
    #                     random_pos_frac_conv[wyckoff_batch],
    #                     torch.ones(
    #                         random_pos_frac_conv[wyckoff_batch].shape[0],
    #                         1,
    #                         device=pos_frac_conv.device,
    #                     ),
    #                 ],
    #                 dim=1,
    #             )
    #             random_pos_frac_proj = (
    #                 torch.einsum("bij,bj->bi", batch.wyckoff_ops, random_pos_tran).squeeze(-1)[
    #                     :, :3
    #                 ]
    #                 % 1.0
    #             ) % 1.0
    #             random_pos_frac_proj = torch.einsum(
    #                 "bij,bj->bi", perm[batch.batch[wyckoff_batch]], random_pos_frac_proj
    #             )
    #             random_pos_cart_proj = torch.einsum(
    #                 "bi,bij->bj",
    #                 random_pos_frac_proj,
    #                 perm_conv_lat_proj[batch.batch[wyckoff_batch]],
    #             )

    #             random_fracs = torch.einsum(
    #                 "bi,bij->bj",
    #                 random_pos_cart_proj,
    #                 prim_lat_inv[batch.batch[wyckoff_batch]],
    #             )
    #             random_fracs = random_fracs % 1.0
    #             random_fracs_diff = random_fracs.unsqueeze(1) - random_fracs.unsqueeze(0)
    #             random_fracs_diff = random_fracs_diff - torch.round(random_fracs_diff)
    #             EPSILON = 5e-4
    #             random_fracs_diff_is_zero = torch.all(
    #                 torch.isclose(
    #                     random_fracs_diff,
    #                     torch.zeros_like(random_fracs_diff),
    #                     rtol=EPSILON,
    #                     atol=EPSILON,
    #                 ),
    #                 dim=-1,
    #             )
    #             random_fracs_idx = random_fracs_diff_is_zero & (
    #                 wyckoff_batch.unsqueeze(0) == wyckoff_batch.unsqueeze(1)
    #             )
    #             random_fracs_idx = ~(random_fracs_idx.triu(diagonal=1).any(dim=0))
    #             # random_fracs_prim = random_fracs[random_fracs_idx]
    #             assert random_fracs_idx.shape[0] == pos_prim_frac_proj_all.shape[0]
    #             pos_prim_frac_proj = pos_prim_frac_proj_all[random_fracs_idx]
    #             if pos_prim_frac_proj.shape[0] == batch.pos.shape[0]:
    #                 self.frac_batch = random_fracs_idx
    #                 break
    #     pos_prim_frac_proj = pos_prim_frac_proj_all[self.frac_batch]

    #     return (pos_prim_frac_proj % 1.0), prim_lat_proj
