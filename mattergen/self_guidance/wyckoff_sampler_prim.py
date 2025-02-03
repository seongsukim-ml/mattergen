from typing import Callable

import torch

from mattergen.diffusion.sampling.pc_sampler import Diffusable
from mattergen.self_guidance.wyckoff_pc_sampler import PredictorCorrector

from mattergen.diffusion.corruption.multi_corruption import apply
from mattergen.diffusion.model_utils import convert_model_out_to_score
from mattergen.self_guidance.symmetry_utils import *
from mattergen.common.utils.data_utils import compute_lattice_polar_decomposition

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
        remove_conditioning_fn: BatchTransform,
        keep_conditioning_fn: BatchTransform | None = None,
        guidance_start: float | None = None,
        guidance_end: float | None = None,
        use_cond_model: False,
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
        # Initialize anchors and uniques
        if self.frac_batch is None or self.frac_batch.shape[0] != x.wyckoff_bat.shape[0]:
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
        assert torch.all(anchors < x.pos.shape[0])

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

        # version 4
        # x.replace(**{"atomic_numbers": x.atomic_numbers[anchors]})

        # def get_unconditional_score():
        #     return super(GuidedPredictorCorrector, self)._score_fn(
        #         x=self._remove_conditioning_fn(x), t=t
        #     )

        def get_unconditional_score():
            x_no_conditioning = self._remove_conditioning_fn(x)
            try:
                pred_data = self.diffusion_module.model(x_no_conditioning, t)
            except:
                import pdb

                pdb.set_trace()

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

            return pred_data.clone().replace(**scores), pred_data

        def get_conditional_score():
            with torch.set_grad_enabled(True):
                x.cell.requires_grad = True
                x.pos.requires_grad = True
                x_no_conditioning = self._remove_conditioning_fn(x)
                try:
                    pred_data = self.diffusion_module.model(x_no_conditioning, t)
                except:
                    import pdb

                    pdb.set_trace()

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
                no_cond = pred_data.clone().detach()

                pred_data = pred_data.replace(**scores)

                anchors = self.anchors
                uniques = self.uniques
                wyckoff_batch = self.wyckoff_batch.to(x.pos.device)
                # x.wyckoff_batch is wrong because it's batch summed over primitive atoms

                self.cf.set_device(x.pos.device)
                scores = {}

                c_mean, c_std = self.diffusion_module.corruption.sdes["cell"].mean_coeff_and_std(
                    x.cell,
                    t,
                    batch=x,
                    batch_idx=self.diffusion_module.corruption._get_batch_indices(x)["cell"],
                )
                c_lim_mean = self.diffusion_module.corruption.sdes["cell"].get_limit_mean(
                    x=x.cell, batch=x
                )
                c_beta = self.diffusion_module.corruption.sdes["cell"].beta(t)
                # prim_clean_lattice = (x.cell - c_std * pred_data.cell) / c_mean  # (B,3,3)
                # prim_clean_lattice = x.cell - c_std * pred_data.cell  # (B,3,3)
                # prim_clean_lattice = (
                #     x.cell - (1 - c_mean) * c_lim_mean - c_std * pred_data.cell
                # ) / c_mean  # (B,3,3)

                ## Assuming that the output is not the epsilon but the score
                # prim_clean_lattice = (x.cell + c_std**2 * pred_data.cell) / c_mean  # (B,3,3)

                prim_clean_lattice = (
                    x.cell - (1 - c_mean) * c_lim_mean + c_std**2 * pred_data.cell
                ) / c_mean  # (B,3,3)

                # prim_clean_lattice = prim_clean_lattice / torch.det(
                #     prim_clean_lattice
                # ).abs().reshape(-1, 1, 1) ** (1.0 / 3)
                # prim_clean_lattice = prim_clean_lattice * c_lim_mean[:, 0, 0].reshape(-1, 1, 1)
                # prim_clean_lattice = compute_lattice_polar_decomposition(prim_clean_lattice)

                conv_clean_lattice = torch.bmm(
                    x.prim_to_conv, prim_clean_lattice
                ).detach()  # (B,3,3)

                conv_clean_lat_vec = self.cf.m2v(self.cf.de_so3(conv_clean_lattice))
                conv_clean_lat_vec_proj = self.cf.proj_k_to_spacegroup(
                    conv_clean_lat_vec, x.space_groups
                )
                conv_clean_lattice_proj = self.cf.v2m(conv_clean_lat_vec_proj)

                # permute
                rank = torch.argsort(
                    -torch.norm(conv_clean_lattice_proj, dim=-1), dim=-1
                )  # (B,3), (rank[0] > rank[1] > rank[2])
                idx = torch.cat([x.space_groups.unsqueeze(-1), rank], dim=-1)

                perm = self.lat_perm[idx[:, 0], idx[:, 1], idx[:, 2], idx[:, 3]]
                perm_conv_clean_lattice_proj = torch.bmm(
                    torch.bmm(perm, conv_clean_lattice_proj), perm.transpose(-1, -2)
                )
                perm_A = self.perm_for_A2[x.space_groups]
                perm_conv_clean_lattice_proj = torch.bmm(
                    torch.bmm(perm_A, perm_conv_clean_lattice_proj), perm_A.transpose(-1, -2)
                )
                prim_clean_lattice_proj = torch.bmm(x.conv_to_prim, perm_conv_clean_lattice_proj)

                # pos: VESDE

                p_mean, p_std = self.diffusion_module.corruption.sdes["pos"].mean_coeff_and_std(
                    x.pos,
                    t,
                    batch=x,
                    batch_idx=self.diffusion_module.corruption._get_batch_indices(x)["pos"],
                )
                # clean_pos = (x.pos - p_std * pred_data.pos) / p_mean
                clean_pos = (x.pos + p_std**2 * pred_data.pos) / p_mean
                clean_pos = (clean_pos % 1) % 1

                # clean_pos_cart = torch.einsum(
                #     "bi,bij->bj", clean_pos.detach(), prim_clean_lattice_proj[x.batch]
                # )  # Which lattice to use? prim_clean? or clean?
                clean_pos_cart = torch.einsum(
                    "bi,bij->bj", clean_pos, prim_clean_lattice[x.batch]
                )  # Which lattice to use? prim_clean? or clean?
                clean_pos_cart = clean_pos_cart.detach()
                clean_pos_frac_conv = (
                    torch.einsum(
                        "bi,bij->bj",
                        clean_pos_cart,
                        torch.linalg.inv(perm_conv_clean_lattice_proj)[x.batch],
                    )
                    % 1.0
                )
                clean_pos_frac_conv = clean_pos_frac_conv % 1.0
                clean_pos_tran = torch.cat(
                    [
                        clean_pos_frac_conv[wyckoff_batch],
                        torch.ones(
                            clean_pos_frac_conv[wyckoff_batch].shape[0], 1, device=clean_pos.device
                        ),
                    ],
                    dim=1,
                )

                clean_pos_frac_proj = (
                    torch.einsum("bij,bj->bi", x.wyckoff_ops, clean_pos_tran).squeeze(-1)[:, :3]
                    % 1.0
                )
                clean_pos_frac_proj = clean_pos_frac_proj % 1.0
                clean_pos_frac_proj = torch.einsum(
                    "bij,bj->bi", perm[x.batch[wyckoff_batch]], clean_pos_frac_proj
                )
                clean_pos_cart_proj = torch.einsum(
                    "bi,bij->bj",
                    clean_pos_frac_proj,
                    perm_conv_clean_lattice_proj[x.batch[wyckoff_batch]],
                )
                prim_lattice_inv = torch.linalg.inv(prim_clean_lattice_proj)

                # Map conv sites to prim sites

                new_fracs = (
                    torch.einsum(
                        "bi,bij->bj", clean_pos_cart_proj, prim_lattice_inv[x.batch[wyckoff_batch]]
                    )
                    % 1.0
                )  # (N,3)
                new_fracs = new_fracs % 1.0
                if self.frac_batch is None or self.frac_batch.shape[0] != wyckoff_batch.shape[0]:
                    self.step = 0
                    new_fracs_diff = new_fracs.unsqueeze(1) - new_fracs.unsqueeze(0)
                    new_fracs_diff = new_fracs_diff - torch.round(new_fracs_diff)
                    EPSILON = 5e-4
                    new_fracs_diff_is_zero = torch.all(
                        torch.isclose(
                            new_fracs_diff,
                            torch.zeros_like(new_fracs_diff),
                            rtol=EPSILON,
                            atol=EPSILON,
                        ),
                        dim=-1,
                    )  # (N,N)
                    new_fracs_idx = new_fracs_diff_is_zero & (
                        wyckoff_batch.unsqueeze(0) == wyckoff_batch.unsqueeze(1)
                    )  # Consider only the same wyckoff site
                    new_fracs_idx = ~(new_fracs_idx.triu(diagonal=1).any(dim=0))
                    new_fracs_prim = new_fracs[new_fracs_idx]

                    # clean_pos_proj = torch.einsum("bij,bj->bi", x.wyckoff_ops_pinv, clean_pos)
                    new_fracs_prim = new_fracs_prim.detach()
                    cur_frac_batch = new_fracs_idx
                    if new_fracs_prim.shape[0] != clean_pos.shape[0]:
                        print(f"{new_fracs_prim.shape[0]} != {clean_pos.shape[0]}")
                        for i in range(5):
                            random_pos_frac_conv = torch.rand_like(clean_pos_frac_conv).to(
                                clean_pos.device
                            )
                            random_pos_tran = torch.cat(
                                [
                                    random_pos_frac_conv[wyckoff_batch],
                                    torch.ones(
                                        random_pos_frac_conv[wyckoff_batch].shape[0],
                                        1,
                                        device=clean_pos.device,
                                    ),
                                ],
                                dim=1,
                            )
                            random_pos_frac_proj = (
                                torch.einsum("bij,bj->bi", x.wyckoff_ops, random_pos_tran).squeeze(
                                    -1
                                )[:, :3]
                                % 1.0
                            ) % 1.0
                            random_pos_frac_proj = torch.einsum(
                                "bij,bj->bi", perm[x.batch[wyckoff_batch]], random_pos_frac_proj
                            )
                            random_pos_cart_proj = torch.einsum(
                                "bi,bij->bj",
                                random_pos_frac_proj,
                                perm_conv_clean_lattice_proj[x.batch[wyckoff_batch]],
                            )

                            random_fracs = torch.einsum(
                                "bi,bij->bj",
                                random_pos_cart_proj,
                                prim_lattice_inv[x.batch[wyckoff_batch]],
                            )
                            random_fracs = random_fracs % 1.0
                            random_fracs_diff = random_fracs.unsqueeze(1) - random_fracs.unsqueeze(
                                0
                            )
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
                            new_fracs_prim = new_fracs[random_fracs_idx]
                            if new_fracs_prim.shape[0] == clean_pos.shape[0]:
                                cur_frac_batch = random_fracs_idx
                                break
                            else:
                                print(f"for {i}: {new_fracs_prim.shape[0]} != {clean_pos.shape[0]}")

                    assert (
                        new_fracs_prim.shape[0] == clean_pos.shape[0]
                    ), f"{new_fracs_prim.shape[0]} != {clean_pos.shape[0]}"
                    if self.frac_batch is not None:
                        print(
                            f"frac_batch is the same {torch.all(self.frac_batch == cur_frac_batch)}"
                        )
                    self.frac_batch = cur_frac_batch.clone()

                new_fracs_prim = new_fracs[self.frac_batch]
                if self.prev_pos_target is None:
                    self.prev_pos_target = new_fracs_prim.detach()

                pos_target = self.prev_pos_target * 0.95 + new_fracs_prim * 0.5
                pos_target = pos_target.detach()
                self.prev_pos_target = pos_target

                # clean_pos_diff = pos_target - clean_pos
                pos_target = pos_target + torch.round(pos_target - clean_pos)
                pos_target = pos_target.detach()

                clean_pos_diff = torch.functional.F.mse_loss(
                    clean_pos, pos_target, reduction="none"
                )

                pos_score = torch.autograd.grad(
                    clean_pos_diff.sum(),
                    x.pos,
                    allow_unused=True,
                    create_graph=True,
                )[0].detach()
                pos_score = -pos_score * 1e-1

                scores["pos"] = pos_score
                x.pos.detach()

                # Invariant under lattice rotation : Too large and has negative when sqrt is applied
                # clean_lattice_proj_invariant = (
                #     torch.bmm(clean_lattice_proj, clean_lattice_proj.transpose(-1, -2))
                #     .flatten(-2, -1)
                #     .sqrt()
                # )
                # clean_latiice_invariant = (
                # torch.bmm(clean_lattice, clean_lattice.transpose(-1, -2)).flatten(-2, -1).sqrt()
                # ) sqrt has Nan since some lattice vectors are negative
                # clean_lattice_proj_invariant = clean_lattice_proj_invariant.detach()
                # clean_lattice_diff = abs((clean_lattice_proj_invariant - clean_latiice_invariant))

                # clean_lattice_diff = abs((clean_lat_vec_proj.detach() - clean_lat_vec))
                # clean_lattice_diff = ((clean_lat_vec_proj.detach() - clean_lat_vec)).norm(dim=-1)
                # lattice_volume = torch.det(prim_clean_lattice_proj).abs().detach()
                # clean_lattice_diff = (
                #     (prim_clean_lattice_proj.detach() - prim_clean_lattice).flatten(-2, -1)
                # ).norm(dim=-1) / (lattice_volume ** (1 / 3))
                prim_clean_lattice_proj_sym = compute_lattice_polar_decomposition(
                    prim_clean_lattice_proj
                )
                # prim_clean_lattice_proj_sym = prim_clean_lattice_proj_sym / torch.det(
                #     prim_clean_lattice_proj
                # ).abs().reshape(-1, 1, 1) ** (1.0 / 3)
                # prim_clean_lattice_proj_sym = prim_clean_lattice_proj_sym * c_lim_mean[
                #     :, 0, 0
                # ].reshape(-1, 1, 1)

                if self.prev_cell_target is None:
                    self.prev_cell_target = prim_clean_lattice_proj_sym.detach()

                cell_target = self.prev_cell_target * 0.9 + prim_clean_lattice_proj_sym * 0.1
                cell_target = cell_target.detach()
                self.prev_cell_target = cell_target

                clean_lattice_diff = (cell_target - prim_clean_lattice).flatten(-2, -1).abs()

                # clean_lattice_diff = torch.functional.F.mse_loss(
                #     prim_clean_lattice_proj.detach(), prim_clean_lattice, reduction="none"
                # )

                # prim_clean_lattice_proj = prim_clean_lattice_proj.detach()
                # clean_lattice_diff = (
                #     torch.cat(lattices_to_params_shape(prim_clean_lattice_proj), dim=-1)
                #     - torch.cat(lattices_to_params_shape(prim_clean_lattice), dim=-1)
                # ).norm(dim=-1)

                lattice_score = torch.autograd.grad(
                    clean_lattice_diff.sum(),
                    x.cell,
                    allow_unused=True,
                )[0].detach()
                # model output directly predict the log p_theta
                # lattice_score2 = lattice_score * 0.5 * c_std * c_mean
                # lattice_score2 = -lattice_score * 0.5 * c_std * c_mean
                lattice_score2 = -lattice_score * 1e-1

                # Check Error
                scores["cell"] = lattice_score2
                x.cell.detach()

                if torch.isnan(scores["cell"]).any() or torch.isnan(scores["pos"]).any():
                    print("Nan in the gradient")
                    scores["cell"][torch.isnan(scores["cell"])] = 0
                if torch.isinf(scores["cell"]).any() or torch.isinf(scores["pos"]).any():
                    raise ValueError("Inf in the gradient")
            # x.cell.requires_grad = False
            # x.pos.requires_grad = False

            # clean_data = {
            #     "pos": clean_pos,
            #     "cell": prim_clean_lattice,
            # }
            # proj_data = {
            #     "pos": new_fracs_prim,
            #     "cell": prim_clean_lattice_proj,
            # }

            # pred_data.replace(atomic_numbers=pred_data.atomic_numbers[anchors])

            if self.step // 2 in [1, 50, 100, 200, 500, 750, 800, 1000]:
                print(f"Step {self.step}")
                print("~~~~~~~~~~~~~~~~~~~~~~")
                print(f"pos score: {pos_score}")
                print(f"pos_pred: {pred_data.pos}")
                print("~~~~~~~~~~~~~~~~~~~~~~")
                print(f"lat score: {lattice_score2}")
                print(f"lat_pred: {pred_data.cell}")
                print("~~~~~~~~~~~~~~~~~~~~~~")
                print(f"clean_pos: {clean_pos}")
                print(f"clean_pos_target: {new_fracs_prim}")
                print(f"pos_target: {pos_target}")
                print(f"pos: {x.pos}")
                print("~~~~~~~~~~~~~~~~~~~~~~")
                print(f"clean_cell: {prim_clean_lattice}")
                print(f"clean_cell_target: {prim_clean_lattice_proj_sym}")
                print(f"cell_target: {cell_target}")
                print(f"cell: {x.cell}")
                # import pdb

                # pdb.set_trace()

            return no_cond, pred_data.replace(**scores)

        _guidance_scale = 0
        # ts = 0.95
        ts = 0.2
        te = 0.0
        if t[0] <= ts and t[0] >= te:
            if self._guidance_start is not None:
                _guidance_scale = self._guidance_start + (
                    self._guidance_end - self._guidance_start
                ) * (ts - t[0])
            else:
                _guidance_scale = self._guidance_scale
        self.step += 1

        # if abs(_guidance_scale - 1) < 1e-15:
        #     # unconditional_score, model_out = get_unconditional_score()
        #     return get_conditional_score()[1]
        if abs(_guidance_scale) < 1e-15:
            return get_unconditional_score()[0]
        else:
            # guided_score = guidance_factor * conditional_score + (1-guidance_factor) * unconditional_score

            # unconditional_score, model_out = get_unconditional_score()
            unconditional_score, conditional_score = get_conditional_score()
            ret = unconditional_score.replace(
                **{
                    k: unconditional_score[k] + _guidance_scale * conditional_score[k]
                    for k in self._multi_corruption.corrupted_fields
                }
            )
            return ret
