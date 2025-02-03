# %%
import pandas as pd
from pymatgen.io.cif import CifParser
from pyxtal import pyxtal
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import numpy as np
from pyxtal.io import write_cif

from pymatgen.core.structure import Structure
from tqdm import tqdm
import pickle
from p_tqdm import p_map
import argparse
import os
from pathlib import Path
import warnings
import json


warnings.filterwarnings("ignore")


def process_cif_to_conventional(cif_str):
    structure = CifParser.from_str(cif_str).get_structures()[0]
    sga = SpacegroupAnalyzer(structure)
    pyx = pyxtal()
    pyx.from_seed(structure, tol=0.01)
    space_group = pyx.group.number
    species = []
    anchors = []
    matrices = []
    # coords = []
    for site in pyx.atom_sites:
        specie = site.specie
        anchor = len(matrices)
        # coord = site.position
        for syms in site.wp:
            species.append(specie)
            matrices.append(syms.affine_matrix)
            # coords.append(syms.operate(coord))
            anchors.append(anchor)
    # anchors = np.array(anchors)
    matrices = np.array(matrices).tolist()
    # coords = np.array(coords) % 1.0
    conv_to_prim = sga.get_conventional_to_primitive_transformation_matrix()
    prim_to_conv = np.linalg.inv(conv_to_prim)
    sym_info = {
        "anchors": anchors,
        "wyckoff_ops": matrices,
        "spacegroup": space_group,
        "conv_to_prim": conv_to_prim.tolist(),
        "prim_to_conv": prim_to_conv.tolist(),
    }
    cif = write_cif(pyx)[805:]
    num_sites = len(species)
    formula = pyx.formula
    return cif, sym_info, num_sites, formula


def process_data(data):
    cif_str = data["cif"]
    cif, sym_info, num_sites, formula = process_cif_to_conventional(cif_str)
    data["cif"] = cif
    data["sym_info"] = sym_info
    data["num_sites"] = num_sites
    data["formula"] = formula
    return data


# main code
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv_path", type=str, default="/home/holywater2/crystal_gen/mattergen/datasets"
    )
    parser.add_argument("--data_name", type=str, default="alex_mp_20")
    parser.add_argument("--mode", type=str, default="val")
    parser.add_argument("--num_cpus", type=int, default=16)
    args = parser.parse_args()
    print("Starting...")
    csv_path = Path(args.csv_path) / args.data_name
    print(f"Processing {csv_path}/{args.mode}.csv")
    df = pd.read_csv(csv_path / f"{args.mode}.csv", index_col=0)
    new_data = p_map(process_data, df.to_dict(orient="records"), num_cpus=args.num_cpus)
    new_df = pd.DataFrame(new_data)

    save_dict = {}
    for i in range(len(new_df)):
        save_dict[new_df["material_id"][i]] = new_df["sym_info"][i]

    with open(csv_path / f"{args.mode}_sym_info.json", "w") as f:
        json.dump(save_dict, f)

    os.makedirs(f"conventional/{args.data_name}", exist_ok=True)
    print(f"Saving to conventional/{args.data_name}/{args.mode}.csv")
    new_df.drop(columns="sym_info").to_csv(f"conventional/{args.data_name}/{args.mode}.csv")
    print("Done!")
