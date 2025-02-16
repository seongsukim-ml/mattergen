import os
from pathlib import Path
from typing import Literal

import fire

from mattergen.common.data.types import TargetProperty
from mattergen.common.utils.eval_utils import MatterGenCheckpointInfo

# from mattergen.generator import CrystalGenerator
from mattergen.self_guidance.wyckoff_generator import CrystalGenerator

import pandas as pd
from pymatgen.io.cif import CifParser
from pyxtal import pyxtal
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import os
from pathlib import Path
import zipfile

import torch

torch.set_num_threads(16)


def main(
    output_path: str,
    model_path: str,
    batch_size: int = 64,
    num_batches: int = 1,
    config_overrides: list[str] | None = None,
    checkpoint_epoch: Literal["best", "last"] | int = "last",
    properties_to_condition_on: TargetProperty | None = None,
    sampling_config_path: str | None = None,
    sampling_config_name: str = "default",
    sampling_config_overrides: list[str] | None = None,
    record_trajectories: bool = True,
    diffusion_guidance_factor: float | None = None,
    strict_checkpoint_loading: bool = True,
    use_wyckoff: bool = True,
    project_save: bool = False,
):
    """
    Evaluate diffusion model against molecular metrics.

    Args:
        model_path: Path to DiffusionLightningModule checkpoint directory.
        output_path: Path to output directory.
        config_overrides: Overrides for the model config, e.g., `model.num_layers=3 model.hidden_dim=128`.
        properties_to_condition_on: Property value to draw conditional sampling with respect to. When this value is an empty dictionary (default), unconditional samples are drawn.
        sampling_config_path: Path to the sampling config file. (default: None, in which case we use `DEFAULT_SAMPLING_CONFIG_PATH` from explorers.common.utils.utils.py)
        sampling_config_name: Name of the sampling config (corresponds to `{sampling_config_path}/{sampling_config_name}.yaml` on disk). (default: default)
        sampling_config_overrides: Overrides for the sampling config, e.g., `condition_loader_partial.batch_size=32`.
        load_epoch: Epoch to load from the checkpoint. If None, the best epoch is loaded. (default: None)
        record: Whether to record the trajectories of the generated structures. (default: True)
        strict_checkpoint_loading: Whether to raise an exception when not all parameters from the checkpoint can be matched to the model.

    NOTE: When specifying dictionary values via the CLI, make sure there is no whitespace between the key and value, e.g., `--properties_to_condition_on={key1:value1}`.
    """
    output_path = Path(output_path).resolve()

    def make_dir(output_path):
        if not os.path.exists(output_path):
            os.makedirs(output_path)

    make_dir(output_path)
    sub_dir_name = "result_0"
    if os.path.exists(output_path / sub_dir_name):
        len_result = len([name for name in os.listdir(output_path) if name.startswith("result")])
        sub_dir_name = f"result_{len_result}"
    output_path = output_path / sub_dir_name
    make_dir(output_path)

    print("OUTPUT_DIR:", output_path, "\n")

    sampling_config_overrides = sampling_config_overrides or []
    config_overrides = config_overrides or []
    properties_to_condition_on = properties_to_condition_on or {}

    checkpoint_info = MatterGenCheckpointInfo(
        model_path=Path(model_path).resolve(),
        load_epoch=checkpoint_epoch,
        config_overrides=config_overrides,
        strict_checkpoint_loading=strict_checkpoint_loading,
    )
    _sampling_config_path = Path(sampling_config_path) if sampling_config_path is not None else None
    generator = CrystalGenerator(
        checkpoint_info=checkpoint_info,
        properties_to_condition_on=properties_to_condition_on,
        batch_size=batch_size,
        num_batches=num_batches,
        sampling_config_name=sampling_config_name,
        sampling_config_path=_sampling_config_path,
        sampling_config_overrides=sampling_config_overrides,
        record_trajectories=record_trajectories,
        diffusion_guidance_factor=(
            diffusion_guidance_factor if diffusion_guidance_factor is not None else 0.0
        ),
        use_wyckoff=use_wyckoff,
    )
    generator.generate(output_dir=Path(output_path), project_to_space_group=project_save)

    # spg = int(output_path.split("/")[-2].split("_")[0].replace("spg", ""))
    spg = int(output_path.parts[-2].split("_")[0].replace("spg", ""))
    print(f"[INFO] Generate CIF files for spg{spg}")
    cif_eval(output_path, spg)
    if project_save:
        if os.path.exists(output_path / "projected"):
            print(f"[INFO] Generate CIF files for spg{spg} with projected")
            cif_eval(output_path / "projected", spg)
        if os.path.exists(output_path / "orig_projected"):
            print(f"[INFO] Generate CIF files for spg{spg} with orig_projected")
            cif_eval(output_path / "orig_projected", spg)

    print("OUTPUT_DIR:", output_path, "\n")


def cif_eval(file_path, spg=None):
    if spg is None:
        # spg = file_path.split("/")[-2].split("_")[0].replace("spg", "")
        spg = file_path.parts[-2].split("_")[0].replace("spg", "")
        spg = int(spg)
    dir_name = "generated_crystals_cif"
    if not os.path.exists(file_path / "generated_crystals_cif.zip"):
        print("No zip file found")
        return

    if os.path.exists(file_path / "generated_crystals_cif"):
        print("Directory already exists")
        # cif_dirs = os.listdir(file_path).glob("generated_crystals_cif")
        cif_dirs = [
            os.path.join(file_path, d)
            for d in os.listdir(file_path)
            if d.startswith("generated_crystals_cif")
        ]
        cif_dirs = [d for d in cif_dirs if os.path.isdir(d)]
        dir_name = f"generated_crystals_cif_{len(cif_dirs)}"

    cif_path = file_path / dir_name
    zipfile.ZipFile(file_path / "generated_crystals_cif.zip", "r").extractall(file_path)
    os.rename(file_path / "tmp", cif_path)
    os.makedirs(file_path / "generated_crystals_cif_spg", exist_ok=True)
    print(cif_path)

    # get list of cifs from directory
    cif_list = [f for f in os.listdir(cif_path) if f.endswith(".cif")]
    cif_list.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))
    cif_list = [os.path.join(cif_path, f) for f in cif_list]

    structures = []
    try:
        for cif_file in cif_list:
            parser = CifParser(cif_file)
            structure = parser.get_structures(primitive=False)[0]
            structures.append(structure)
    except Exception as e:
        print(e)
        print("Failed to parse CIF files")
        return

    def save_res(tol=0.1):
        print(f"[I] save the symprec={tol}")
        res = {}
        for i, structure in enumerate(structures):
            try:
                sgn = SpacegroupAnalyzer(structure, symprec=tol).get_space_group_number()
            except:
                sgn = 0
            # print(i,sgn)
            if sgn == spg:
                structure.to(
                    filename=cif_path.with_name("generated_crystals_cif_spg") / f"spg{spg}_{i}.cif"
                )
            if sgn not in res:
                res[sgn] = 0
            res[sgn] += 1
        for k in sorted(res.keys()):
            print(f"{k:3} {res[k]}  {res[k]/len(structures):.2f}")
        print()

        if spg in res:
            print(f"{spg} {res[spg]}  {res[spg]/len(structures):.2f}")
        else:
            print(f"spg{spg} is not found")
            res[spg] = 0

        # write the number of spg
        with open(
            file_path / f"symprec_{tol:.2f}_spg{spg}_{res[spg]}_{len(structures)}.txt", "w"
        ) as f:
            for k in sorted(res.keys()):
                f.write(f"{k:3} {res[k]}  {res[k]/len(structures):.2f}\n")
            f.write("\n")
            if spg in res:
                f.write(f"{spg} {res[spg]}  {res[spg]/len(structures):.2f}\n")
            else:
                f.write(f"spg{spg} is not found\n")
        print("[I] Save done")
        print("")

    try:
        save_res(0.01)
    except Exception as e:
        print(e)
        print("Failed to save with symprec=0.01")
    try:
        save_res(0.05)
    except Exception as e:
        print(e)
        print("Failed to save with symprec=0.05")
    try:
        save_res(0.1)
    except Exception as e:
        print(e)
        print("Failed to save with symprec=0.1")


if __name__ == "__main__":
    # use fire instead of argparse to allow for the specification of dictionary values via the CLI
    fire.Fire(main)
