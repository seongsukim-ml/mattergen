from __future__ import annotations

from typing import Generic, Mapping, Tuple, TypeVar

import torch
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


class PredictorCorrector(Generic[Diffusable]):
    """Generates samples using predictor-corrector sampling."""

    def __init__(
        self,
        *,
        diffusion_module: DiffusionModule,
        predictor_partials: dict[str, PredictorPartial] | None = None,
        corrector_partials: dict[str, CorrectorPartial] | None = None,
        device: torch.device,
        n_steps_corrector: int,
        N: int,
        eps_t: float = 1e-3,
        max_t: float | None = None,
    ):
        """
        Args:
            diffusion_module: diffusion module
            predictor_partials: partials for constructing predictors. Keys are the names of the corruptions.
            corrector_partials: partials for constructing correctors. Keys are the names of the corruptions.
            device: device to run on
            n_steps_corrector: number of corrector steps
            N: number of noise levels
            eps_t: diffusion time to stop denoising at
            max_t: diffusion time to start denoising at. If None, defaults to the maximum diffusion time. You may want to start at T-0.01, say, for numerical stability.
        """
        self._diffusion_module = diffusion_module
        self.N = N

        if max_t is None:
            max_t = self._multi_corruption.T
        assert max_t <= self._multi_corruption.T, "Denoising cannot start from beyond T"

        self._max_t = max_t
        assert (
            corrector_partials or predictor_partials
        ), "Must specify at least one predictor or corrector"
        corrector_partials = corrector_partials or {}
        predictor_partials = predictor_partials or {}
        if self._multi_corruption.discrete_corruptions:
            # These all have property 'N' because they are D3PM type
            assert set(c.N for c in self._multi_corruption.discrete_corruptions.values()) == {N}  # type: ignore

        self._predictors = {
            k: v(corruption=self._multi_corruption.corruptions[k], score_fn=None)
            for k, v in predictor_partials.items()
        }

        self._correctors = {
            k: v(
                corruption=self._multi_corruption.corruptions[k],
                n_steps=n_steps_corrector,
                score_fn=None,
            )
            for k, v in corrector_partials.items()
        }
        self._eps_t = eps_t
        self._n_steps_corrector = n_steps_corrector
        self._device = device

    @property
    def diffusion_module(self) -> DiffusionModule:
        return self._diffusion_module

    @property
    def _multi_corruption(self) -> MultiCorruption:
        return self._diffusion_module.corruption

    def _score_fn(self, x: Diffusable, t: torch.Tensor) -> Diffusable:
        return self._diffusion_module.score_fn(x, t)

    @classmethod
    def from_pl_module(cls, pl_module: DiffusionLightningModule, **kwargs) -> PredictorCorrector:
        return cls(diffusion_module=pl_module.diffusion_module, device=pl_module.device, **kwargs)

    @torch.no_grad()
    def sample(
        self, conditioning_data: BatchedData, mask: Mapping[str, torch.Tensor] | None = None
    ) -> SampleAndMean:
        """Create one sample for each of a batch of conditions.
        Args:
            conditioning_data: batched conditioning data. Even if you think you don't want conditioning, you still need to pass a batch of conditions
               because the sampler uses these to determine the shapes of things to generate.
            mask: for inpainting. Keys should be a subset of the keys in `data`. 1 indicates data that should be fixed, 0 indicates data that should be replaced with sampled values.
                Shapes of values in `mask` must match the shapes of values in `conditioning_data`.
        Returns:
           (batch, mean_batch). The difference between these is that `mean_batch` has no noise added at the final denoising step.

        """
        return self._sample_maybe_record(conditioning_data, mask=mask, record=False)[:2]

    @torch.no_grad()
    def sample_with_record(
        self, conditioning_data: BatchedData, mask: Mapping[str, torch.Tensor] | None = None
    ) -> SampleAndMeanAndRecords:
        """Create one sample for each of a batch of conditions.
        Args:
            conditioning_data: batched conditioning data. Even if you think you don't want conditioning, you still need to pass a batch of conditions
               because the sampler uses these to determine the shapes of things to generate.
            mask: for inpainting. Keys should be a subset of the keys in `data`. 1 indicates data that should be fixed, 0 indicates data that should be replaced with sampled values.
                Shapes of values in `mask` must match the shapes of values in `conditioning_data`.
        Returns:
           (batch, mean_batch). The difference between these is that `mean_batch` has no noise added at the final denoising step.

        """
        return self._sample_maybe_record(conditioning_data, mask=mask, record=True)

    @torch.no_grad()
    def _sample_maybe_record(
        self,
        conditioning_data: BatchedData,
        mask: Mapping[str, torch.Tensor] | None = None,
        record: bool = False,
    ) -> SampleAndMeanAndMaybeRecords:
        """Create one sample for each of a batch of conditions.
        Args:
            conditioning_data: batched conditioning data. Even if you think you don't want conditioning, you still need to pass a batch of conditions
               because the sampler uses these to determine the shapes of things to generate.
            mask: for inpainting. Keys should be a subset of the keys in `data`. 1 indicates data that should be fixed, 0 indicates data that should be replaced with sampled values.
                Shapes of values in `mask` must match the shapes of values in `conditioning_data`.
        Returns:
           (batch, mean_batch, recorded_samples, recorded_predictions).
           The difference between the former two is that `mean_batch` has no noise added at the final denoising step.
           The latter two are only returned if `record` is True, and contain the samples and predictions from each step of the diffusion process.

        """
        if isinstance(self._diffusion_module, torch.nn.Module):
            self._diffusion_module.eval()
        mask = mask or {}
        conditioning_data = conditioning_data.to(self._device)
        mask = {k: v.to(self._device) for k, v in mask.items()}
        batch = _sample_prior(self._multi_corruption, conditioning_data, mask=mask)
        return self._denoise(batch=batch, mask=mask, record=record)

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

        # Decreasing timesteps from T to eps_t
        timesteps = torch.linspace(self._max_t, self._eps_t, self.N, device=self._device)
        dt = -torch.tensor((self._max_t - self._eps_t) / (self.N - 1)).to(self._device)

        for i in tqdm(range(self.N), miniters=50, mininterval=5):
            # Set the timestep
            t = torch.full((batch.get_batch_size(),), timesteps[i], device=self._device)

            # Corrector updates.
            if self._correctors:
                for _ in range(self._n_steps_corrector):
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

            # Predictor updates
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
        return batch, mean_batch, recorded_samples


def _project_to_space_group(batch):
    cf = CrystalFamily()
    cf.set_device(batch.pos.device)
    lat_perm, perm_for_A1, perm_for_A2 = get_latttice_permutations(device=batch.pos.device)

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
    perm_A = perm_for_A2[batch.space_groups]

    perm_conv_lat_proj = torch.bmm(torch.bmm(perm, conv_lat_proj), perm.transpose(-1, -2))
    perm_conv_lat_proj = torch.bmm(torch.bmm(perm_A, perm_conv_lat_proj), perm_A.transpose(-1, -2))

    prim_lat_proj = torch.bmm(batch.conv_to_prim, perm_conv_lat_proj)

    pos_cart = torch.einsum("bi,bij->bj", batch.pos, batch.cell[batch.batch])
    pos_frac_conv = torch.einsum(
        "bi,bij->bj", pos_cart, torch.inverse(perm_conv_lat_proj)[batch.batch]
    )
    pos_tran = torch.cat(
        [
            pos_frac_conv[wyckoff_batch],
            torch.ones(pos_frac_conv[wyckoff_batch].shape[0], 1, device=batch.pos.device),
        ],
        dim=1,
    )

    pos_frac_proj = torch.einsum("bij,bj->bi", batch.wyckoff_ops, pos_tran).squeeze(-1)[:, :3] % 1.0
    pos_frac_proj = torch.einsum("bij,bj->bi", perm[batch.batch[wyckoff_batch]], pos_frac_proj)
    pos_cart_porj = torch.einsum(
        "bi,bij->bj", pos_frac_proj, perm_conv_lat_proj[batch.batch[wyckoff_batch]]
    )

    prim_lat_inv = torch.inverse(prim_lat_proj)
    pos_prim_frac_proj_all = torch.einsum(
        "bi,bij->bj", pos_cart_porj, prim_lat_inv[batch.batch[wyckoff_batch]]
    )

    ## Get prim idx
    for i in range(5):
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
            torch.einsum("bij,bj->bi", batch.wyckoff_ops, random_pos_tran).squeeze(-1)[:, :3] % 1.0
        ) % 1.0
        random_pos_frac_proj = torch.einsum(
            "bij,bj->bi", perm[batch.batch[wyckoff_batch]], random_pos_frac_proj
        )
        random_pos_cart_proj = torch.einsum(
            "bi,bij->bj",
            random_pos_frac_proj,
            conv_lat_proj[batch.batch[wyckoff_batch]],
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
            break

    return batch.clone().replace(pos=pos_prim_frac_proj, cell=prim_lat_proj)


def _mask_replace(
    samples_means: dict[str, Tuple[torch.Tensor, torch.Tensor]],
    batch: BatchedData,
    mean_batch: BatchedData,
    mask: dict[str, torch.Tensor | None],
) -> SampleAndMean:
    # Apply masks
    samples_means = apply(
        fns={k: _mask_both for k in samples_means},
        broadcast={},
        sample_and_mean=samples_means,
        mask=mask,
        old_x=batch,
    )

    # Put the updated values in `batch` and `mean_batch`
    batch = batch.replace(**{k: v[0] for k, v in samples_means.items()})
    mean_batch = mean_batch.replace(**{k: v[1] for k, v in samples_means.items()})
    return batch, mean_batch


def _mask_both(
    *, sample_and_mean: Tuple[torch.Tensor, torch.Tensor], old_x: torch.Tensor, mask: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    return tuple(_mask(old_x=old_x, new_x=x, mask=mask) for x in sample_and_mean)  # type: ignore


def _mask(*, old_x: torch.Tensor, new_x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """Replace new_x with old_x where mask is 1."""
    if mask is None:
        return new_x
    else:
        return new_x.lerp(old_x, mask)


def _sample_prior(
    multi_corruption: MultiCorruption,
    conditioning_data: BatchedData,
    mask: Mapping[str, torch.Tensor] | None,
) -> BatchedData:
    samples = {
        k: multi_corruption.corruptions[k]
        .prior_sampling(
            shape=conditioning_data[k].shape,
            conditioning_data=conditioning_data,
            batch_idx=conditioning_data.get_batch_idx(field_name=k),
        )
        .to(conditioning_data[k].device)
        for k in multi_corruption.corruptions
    }
    mask = mask or {}
    for k, msk in mask.items():
        if k in multi_corruption.corrupted_fields:
            samples[k].lerp_(conditioning_data[k], msk)
    return conditioning_data.replace(**samples)
