import io
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from zipfile import ZipFile

import ase.io
import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from tqdm import tqdm

from mattergen.common.data.chemgraph import ChemGraph
from mattergen.common.data.collate import collate
from mattergen.common.data.condition_factory import ConditionLoader
from mattergen.common.data.num_atoms_distribution import NUM_ATOMS_DISTRIBUTIONS
from mattergen.common.data.types import TargetProperty
from mattergen.common.utils.data_utils import lattice_matrix_to_params_torch
from mattergen.common.utils.eval_utils import (
    MatterGenCheckpointInfo,
    get_crystals_list,
    load_model_diffusion,
    make_structure,
    save_structures,
)
from mattergen.common.utils.globals import DEFAULT_SAMPLING_CONFIG_PATH
from mattergen.diffusion.lightning_module import DiffusionLightningModule
from mattergen.self_guidance.symmetry_utils import *

# from mattergen.diffusion.sampling.pc_sampler import PredictorCorrector
from mattergen.self_guidance.wyckoff_pc_sampler import PredictorCorrector
import json


def draw_samples_from_sampler(
    sampler: PredictorCorrector,
    condition_loader: ConditionLoader,
    properties_to_condition_on: TargetProperty | None = None,
    output_path: Path | None = None,
    cfg: DictConfig | None = None,
    record_trajectories: bool = True,
    project_to_space_group: bool = False,
) -> list[Structure]:

    # Dict
    properties_to_condition_on = properties_to_condition_on or {}

    # we cannot conditional sample on something on which the model was not trained to condition on
    assert all([key in sampler.diffusion_module.model.cond_fields_model_was_trained_on for key in properties_to_condition_on.keys()])  # type: ignore

    all_samples_list = []
    all_trajs_list = []
    all_projected_list = []
    all_projected_orig_list = []

    for conditioning_data, mask in tqdm(condition_loader, desc="Generating samples"):
        # generate samples
        if record_trajectories:
            sample, mean, intermediate_samples, projected_mean, projected_orig = (
                sampler.sample_with_record(conditioning_data, mask)
            )
            all_trajs_list.extend(list_of_time_steps_to_list_of_trajectories(intermediate_samples))
            projected = projected_mean
        else:
            sample, mean = sampler.sample(conditioning_data, mask)
        all_samples_list.extend(mean.to_data_list())
        all_projected_orig_list.extend(projected_orig.to_data_list())
        if project_to_space_group:
            # projected = _project_to_space_group(sample)
            all_projected_list.extend(projected.to_data_list())
    all_samples = collate(all_samples_list)
    assert isinstance(all_samples, ChemGraph)
    lengths, angles = lattice_matrix_to_params_torch(all_samples.cell)
    all_samples = all_samples.replace(lengths=lengths, angles=angles)

    generated_strucs = structure_from_model_output(
        all_samples["pos"].reshape(-1, 3),
        all_samples["atomic_numbers"].reshape(-1),
        all_samples["lengths"].reshape(-1, 3),
        all_samples["angles"].reshape(-1, 3),
        all_samples["num_atoms"].reshape(-1),
    )

    if project_to_space_group:
        all_projected = collate(all_projected_list)
        assert isinstance(all_projected, ChemGraph)
        lengths1, angles1 = lattice_matrix_to_params_torch(all_projected.cell)
        all_projected = all_projected.replace(lengths=lengths1, angles=angles1)

        projected_strucs = structure_from_model_output(
            all_projected["pos"].reshape(-1, 3),
            all_projected["atomic_numbers"].reshape(-1),
            all_projected["lengths"].reshape(-1, 3),
            all_projected["angles"].reshape(-1, 3),
            all_projected["num_atoms"].reshape(-1),
        )

        all_projected_orig = collate(all_projected_orig_list)
        assert isinstance(all_projected_orig, ChemGraph)
        lengths2, angles2 = lattice_matrix_to_params_torch(all_projected_orig.cell)
        all_projected_orig = all_projected_orig.replace(lengths=lengths2, angles=angles2)

        projected_orig_strucs = structure_from_model_output(
            all_projected_orig["pos"].reshape(-1, 3),
            all_projected_orig["atomic_numbers"].reshape(-1),
            all_projected_orig["lengths"].reshape(-1, 3),
            all_projected_orig["angles"].reshape(-1, 3),
            all_projected_orig["num_atoms"].reshape(-1),
        )

    if output_path is not None:
        assert cfg is not None
        # Save structures to disk in both a extxyz file and a compressed zip file.
        # do this before uploading to mongo in case there is an authentication error
        save_structures(output_path, generated_strucs)

        if project_to_space_group:
            os.makedirs(output_path / "projected", exist_ok=True)
            save_structures(output_path / "projected", projected_strucs)
            os.makedirs(output_path / "orig_projected", exist_ok=True)
            save_structures(output_path / "orig_projected", projected_orig_strucs)

        if record_trajectories:
            dump_trajectories(
                output_path=output_path,
                all_trajs_list=all_trajs_list,
            )
        # save structure ids
        with open(output_path / "structure_ids.json", "w") as f:
            json.dump(condition_loader.dataset.structure_id, f)

    return generated_strucs


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

    anchors = batch.anchors.clone().detach()
    idx = 0
    cum = 0
    for len, num_atom in zip(batch.anchors_len, batch.num_atoms):
        anchors[idx : idx + len] = anchors[idx : idx + len] + cum
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
    pos_cart_proj = torch.einsum(
        "bi,bij->bj", pos_frac_proj, perm_conv_lat_proj[batch.batch[wyckoff_batch]]
    )

    prim_lat_inv = torch.inverse(prim_lat_proj)
    pos_prim_frac_proj_all = (
        torch.einsum("bi,bij->bj", pos_cart_proj, prim_lat_inv[batch.batch[wyckoff_batch]]) % 1.0
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
            perm_conv_lat_proj[batch.batch[wyckoff_batch]],
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

    assert pos_prim_frac_proj.shape[0] == batch.pos.shape[0]

    return batch.clone().replace(
        pos=pos_prim_frac_proj, cell=prim_lat_proj, atomic_numbers=batch.atomic_numbers[anchors]
    )


def list_of_time_steps_to_list_of_trajectories(
    list_of_time_steps: list[ChemGraph],
) -> list[list[ChemGraph]]:
    # Rearrange the shapes of the recorded intermediate samples and predictions
    # We get a list of <num_timesteps> many ChemGraphBatches, each containing <batch_size>
    # many ChemGraphs. Instead, we group all the ChemGraphs of the same trajectory together,
    # i.e., we construct lists of <batch_size> many lists of
    # <num_timesteps * (1 + num_corrector_steps)> many ChemGraphs.

    # <num_timesteps * (1 + num_corrector_steps)> many lists of <batch_size> many ChemGraphs
    data_lists_per_timesteps = [x.to_data_list() for x in list_of_time_steps]

    # <batch_size> many lists of <num_timesteps * (1 + num_corrector_steps)> many ChemGraphs.
    data_lists_per_sample = [
        [data_lists_per_timesteps[ix_t][ix_traj] for ix_t in range(len(data_lists_per_timesteps))]
        for ix_traj in range(len(data_lists_per_timesteps[0]))
    ]
    return data_lists_per_sample


def dump_trajectories(
    output_path: Path,
    all_trajs_list: list[list[ChemGraph]],
) -> None:
    try:
        # We gather all trajectories in a single zip file as .extxyz files.
        # This way we can view them easily after downloading.
        with ZipFile(output_path / "generated_trajectories.zip", "w") as zip_obj:
            for ix, traj in enumerate(all_trajs_list):
                strucs = structures_from_trajectory(traj)
                ase_atoms = [AseAtomsAdaptor.get_atoms(crystal) for crystal in strucs]
                str_io = io.StringIO()
                ase.io.write(str_io, ase_atoms, format="extxyz")
                str_io.flush()
                zip_obj.writestr(f"gen_{ix}.extxyz", str_io.getvalue())
    except IOError as e:
        print(f"Got error {e} writing the trajectory to disk.")
    except ValueError as e:
        print(f"Got error ValueError '{e}' writing the trajectory to disk.")


def structure_from_model_output(
    frac_coords, atom_types, lengths, angles, num_atoms
) -> list[Structure]:
    structures = [
        make_structure(
            lengths=d["lengths"],
            angles=d["angles"],
            atom_types=d["atom_types"],
            frac_coords=d["frac_coords"],
        )
        for d in get_crystals_list(
            frac_coords.cpu(),
            atom_types.cpu(),
            lengths.cpu(),
            angles.cpu(),
            num_atoms.cpu(),
        )
    ]
    return structures


def structures_from_trajectory(traj: list[ChemGraph]) -> list[Structure]:
    all_strucs = []
    for batch in traj:
        cell = batch.cell
        lengths, angles = lattice_matrix_to_params_torch(cell)
        all_strucs.extend(
            structure_from_model_output(
                frac_coords=batch.pos,
                atom_types=batch.atomic_numbers,
                lengths=lengths,
                angles=angles,
                num_atoms=batch.num_atoms,
            )
        )

    return all_strucs


@dataclass
class CrystalGenerator:
    checkpoint_info: MatterGenCheckpointInfo

    # These may be set at runtime
    batch_size: int | None = None
    num_batches: int | None = None
    target_compositions_dict: list[dict[str, float]] | None = None
    num_atoms_distribution: str = "ALEX_MP_20"

    # Conditional generation
    diffusion_guidance_factor: float = 0.0
    properties_to_condition_on: TargetProperty | None = None

    # Additional overrides, only has an effect when using a diffusion-codebase model
    sampling_config_overrides: list[str] | None = None

    # These only have an effect when using a legacy model
    num_samples_per_batch: int = 1
    niggli_reduction: bool = False

    # Config path, if None will default to DEFAULT_SAMPLING_CONFIG_PATH
    sampling_config_path: Path | None = None
    sampling_config_name: str = "default"

    record_trajectories: bool = True  # store all intermediate samples by default

    use_wyckoff: bool = False

    # These attributes are set when prepare() method is called.
    _model: DiffusionLightningModule | None = None
    _cfg: DictConfig | None = None

    def __post_init__(self) -> None:
        assert self.num_atoms_distribution in NUM_ATOMS_DISTRIBUTIONS, (
            f"num_atoms_distribution must be one of {list(NUM_ATOMS_DISTRIBUTIONS.keys())}, "
            f"but got {self.num_atoms_distribution}. To add your own distribution, "
            "please add it to mattergen.common.data.num_atoms_distribution.NUM_ATOMS_DISTRIBUTIONS."
        )

    @property
    def model(self) -> DiffusionLightningModule:
        self.prepare()
        assert self._model is not None
        return self._model

    @property
    def cfg(self) -> DictConfig:
        self._cfg = self.checkpoint_info.config
        assert self._cfg is not None
        return self._cfg

    @property
    def num_structures_to_generate(self) -> int:
        """Returns the total number of structures to generate if `batch_size` and `num_batches` are specified at construction time;
        otherwise, raises an AssertionError.
        """
        assert self.batch_size is not None
        assert self.num_batches is not None
        return self.batch_size * self.num_batches

    @property
    def sampling_config(self) -> DictConfig:
        """Returns the sampling config if `batch_size` and `num_batches` are specified at construction time;
        otherwise, raises an AssertionError.
        """
        assert self.batch_size is not None
        assert self.num_batches is not None
        return self.load_sampling_config(
            batch_size=self.batch_size,
            num_batches=self.num_batches,
            target_compositions_dict=self.target_compositions_dict,
        )

    def get_condition_loader(
        self,
        sampling_config: DictConfig,
        target_compositions_dict: list[dict[str, float]] | None = None,
    ) -> ConditionLoader:
        condition_loader_partial = instantiate(sampling_config.condition_loader_partial)
        if target_compositions_dict is None:
            return condition_loader_partial(properties=self.properties_to_condition_on)

        return condition_loader_partial(target_compositions_dict=target_compositions_dict)

    def load_sampling_config(
        self,
        batch_size: int,
        num_batches: int,
        target_compositions_dict: list[dict[str, float]] | None = None,
    ) -> DictConfig:
        """
        Create a sampling config from the given parameters.
        We specify certain sampling hyperparameters via the sampling config that is loaded via hydra.
        """
        if self.sampling_config_overrides is None:
            sampling_config_overrides = []
        else:
            # avoid modifying the original list
            sampling_config_overrides = self.sampling_config_overrides.copy()

        if self.use_wyckoff:
            sampling_config_overrides += [
                f"+condition_loader_partial.batch_size={batch_size}",
                f"+condition_loader_partial.num_samples={num_batches * batch_size}",
                # f"sampler_partial.guidance_scale={self.diffusion_guidance_factor}",
            ]
        else:
            if target_compositions_dict is None:
                # Default `condition_loader_partial` is
                # mattergen.common.data.condition_factory.get_number_of_atoms_condition_loader
                sampling_config_overrides += [
                    f"+condition_loader_partial.num_atoms_distribution={self.num_atoms_distribution}",
                    f"+condition_loader_partial.batch_size={batch_size}",
                    f"+condition_loader_partial.num_samples={num_batches * batch_size}",
                    f"sampler_partial.guidance_scale={self.diffusion_guidance_factor}",
                ]
            else:
                # `condition_loader_partial` for fixed atom type (crystal structure prediction)
                num_structures_to_generate_per_composition = (
                    num_batches * batch_size // len(target_compositions_dict)
                )
                sampling_config_overrides += [
                    "condition_loader_partial._target_=mattergen.common.data.condition_factory.get_composition_data_loader",
                    f"+condition_loader_partial.num_structures_to_generate_per_composition={num_structures_to_generate_per_composition}",
                    f"+condition_loader_partial.batch_size={batch_size}",
                ]
        return self._load_sampling_config(
            sampling_config_overrides=sampling_config_overrides,
            sampling_config_path=self.sampling_config_path,
            sampling_config_name=self.sampling_config_name,
        )

    def _load_sampling_config(
        self,
        sampling_config_path: Path | None = None,
        sampling_config_name: str = "default",
        sampling_config_overrides: list[str] | None = None,
    ) -> DictConfig:
        if sampling_config_path is None:
            sampling_config_path = DEFAULT_SAMPLING_CONFIG_PATH

        if sampling_config_overrides is None:
            sampling_config_overrides = []

        with hydra.initialize_config_dir(os.path.abspath(str(sampling_config_path))):
            sampling_config = hydra.compose(
                config_name=sampling_config_name, overrides=sampling_config_overrides
            )
        return sampling_config

    def prepare(self) -> None:
        """Loads the model from checkpoint and prepares for generation."""
        if self._model is not None:
            return
        model = load_model_diffusion(self.checkpoint_info)
        model = model.to("cuda" if torch.cuda.is_available() else "cpu")
        self._model = model
        self._cfg = self.checkpoint_info.config

    def generate(
        self,
        batch_size: int | None = None,
        num_batches: int | None = None,
        target_compositions_dict: list[dict[str, float]] | None = None,
        output_dir: str = "outputs",
        project_to_space_group: bool = False,
    ) -> list[Structure]:
        # Prioritize the runtime provided batch_size, num_batches and target_compositions_dict
        batch_size = batch_size or self.batch_size
        num_batches = num_batches or self.num_batches
        target_compositions_dict = target_compositions_dict or self.target_compositions_dict
        assert batch_size is not None
        assert num_batches is not None

        # print config for debugging and reproducibility
        # print("\nModel config:")
        # print(OmegaConf.to_yaml(self.cfg, resolve=True))

        sampling_config = self.load_sampling_config(
            batch_size=batch_size,
            num_batches=num_batches,
            target_compositions_dict=target_compositions_dict,
        )

        # print("\nSampling config:")
        # print(OmegaConf.to_yaml(sampling_config, resolve=True))

        condition_loader = self.get_condition_loader(sampling_config, target_compositions_dict)

        sampler_partial = instantiate(sampling_config.sampler_partial)
        sampler = sampler_partial(pl_module=self.model)

        generated_structures = draw_samples_from_sampler(
            sampler=sampler,
            condition_loader=condition_loader,
            cfg=self.cfg,
            output_path=Path(output_dir),
            properties_to_condition_on=self.properties_to_condition_on,
            record_trajectories=self.record_trajectories,
            project_to_space_group=project_to_space_group,
        )

        return generated_structures
