import os
import json
import numpy as np
import pandas as pd
from pathlib import Path

from pymatgen.core import Lattice
from pymatgen.core.structure import Structure
from pymatgen.io.cif import CifParser
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from scipy.optimize import linear_sum_assignment

from pyxtal import pyxtal
from pyxtal.io import write_cif
from pyxtal.symmetry import Wyckoff_position

from tqdm import tqdm
from p_tqdm import p_map
from functools import partial

import zipfile
import spglib
import warnings

import numpy.random as random
from constant import lattice_types, symbols, symbols0


# warnings.filterwarnings("ignore")

# mulit_cpu examples:
# for "func(a,b,c)"
# p_map(partial(c=C),a,b,num_cpus)

import sys

sys.path.append("/home/holywater2/crystal_gen/mattergen/mattergen/self_guidance")
import symmetry_utils
import torch


lat_perm = np.zeros((231, 3, 3, 3, 3, 3))
lat_perm[:, :, :, :, 0, 0] = 1
lat_perm[:, :, :, :, 1, 1] = 1
lat_perm[:, :, :, :, 2, 2] = 1
# 1, 0, 2 implies lat(1) > lat(0) > lat(2)
possible_perms = [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]]

perm_for_A1 = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
perm_for_A2 = np.eye(3).reshape(1, 3, 3).repeat(231, axis=0)

# orthorhombic with A:
for i in [38, 39, 40, 41]:
    lat_perm[i, 0, 1, 2] = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
    lat_perm[i, 0, 2, 1] = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
    lat_perm[i, 1, 0, 2] = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    lat_perm[i, 1, 2, 0] = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
    lat_perm[i, 2, 0, 1] = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
    lat_perm[i, 2, 1, 0] = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    for p in possible_perms:
        lat_perm[i, p[0], p[1], p[2]] = perm_for_A1 @ lat_perm[0, p[0], p[1], p[2]]
    perm_for_A2[i] = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])


def load_csv(
    csv_dir: str = "/home/holywater2/crystal_gen/mattergen/datasets",
    dataset_name: str = "mp_20",
    dataset_mode: str = "val",
) -> pd.DataFrame:
    csv_path = f"{csv_dir}/{dataset_name}/{dataset_mode}.csv"
    return pd.read_csv(csv_path)


def load_cifs(
    csv_dir: str = "/home/holywater2/crystal_gen/mattergen/datasets",
    dataset_name: str = "mp_20",
    dataset_mode: str = "val",
):
    df = load_csv(csv_dir, dataset_name, dataset_mode)
    cifs = df["cif"].tolist()
    return cifs


spglib.SpaceGroupType


def structure_from_cif(cif_str: str):
    if cif_str.endswith(".cif"):
        structure = CifParser(cif_str).get_structures()[0]
    else:
        structure = CifParser.from_str(cif_str).get_structures()[0]
    return structure


def cif_list_to_structures(cif_list):
    structures = []
    for cif_str in cif_list:
        structure = structure_from_cif(cif_str)
        structures.append(structure)
    return structures


def get_symmetry_info(crystal, tol=0.01):
    spga = SpacegroupAnalyzer(crystal, symprec=tol)
    crystal = spga.get_refined_structure()
    c = pyxtal()
    try:
        c.from_seed(crystal, tol=0.01)
    except:
        c.from_seed(crystal, tol=0.0001)
    space_group = c.group.number
    species = []
    anchors = []
    matrices = []
    coords = []
    for site in c.atom_sites:
        specie = site.specie
        anchor = len(matrices)
        coord = site.position
        for syms in site.wp:
            species.append(specie)
            matrices.append(syms.affine_matrix)
            coords.append(syms.operate(coord))
            anchors.append(anchor)
    anchors = np.array(anchors)
    matrices = np.array(matrices)
    coords = np.array(coords) % 1.0
    sym_info = {"anchors": anchors, "wyckoff_ops": matrices, "spacegroup": space_group}
    crystal = Structure(
        lattice=Lattice.from_parameters(*np.array(c.lattice.get_para(degree=True))),
        species=species,
        coords=coords,
        coords_are_cartesian=False,
    )
    return crystal, sym_info


def get_symmetry_info_from_pri_to_conv(crystal, tol=0.01):
    spga = SpacegroupAnalyzer(crystal, symprec=tol)
    spg_num = spga.get_space_group_number()
    symmetry_dataset = spga.get_symmetry_dataset()
    lat_conv_to_pri = symmetry_dataset.transformation_matrix
    lat_pri_to_conv = np.linalg.inv(lat_conv_to_pri)
    equivalent_atoms = symmetry_dataset.equivalent_atoms
    conv_mapping_to_primitive = symmetry_dataset.std_mapping_to_primitive

    matrices = []

    for periodic_site in np.unique(equivalent_atoms):
        wp = Wyckoff_position.from_group_and_letter(
            spg_num, symmetry_dataset.wyckoffs[periodic_site]
        )
        for syms in wp:
            pass


def test_structures(N=100, print_warn=False, dataset_name="mp_20", dataset_mode="val", num_cpus=32):
    if not print_warn:
        warnings.filterwarnings("ignore")
    cifs = load_cifs(dataset_name=dataset_name, dataset_mode=dataset_mode)
    structures = []
    if N == -1:
        N = len(cifs)
    structures = p_map(structure_from_cif, cifs[:N], num_cpus=num_cpus)

    if not print_warn:
        warnings.filterwarnings("default")
    print(f"[I] Loaded {len(structures)} structures")
    return structures


def load_sym_info(
    json_dir: str = "/home/holywater2/crystal_gen/mattergen/_my_scripts/space_group_info",
    dataset_name: str = "mp_20",
    dataset_mode: str = "val",
):
    assert dataset_mode in ["train", "val"]
    ret = {}
    for file_name in ["sg_counts", "sym_info"]:
        json_path = f"{json_dir}/{dataset_name}_{dataset_mode}/{file_name}.json"
        with open(json_path, "r") as f:
            ret[file_name] = json.load(f)
    ret["sg_counts"] = dict(sorted(ret["sg_counts"].items(), key=lambda x: int(x[1]), reverse=True))
    return ret


def process_cif_to_conventional(cif_str):
    structure = CifParser.from_str(cif_str).get_structures()[0]
    sga = SpacegroupAnalyzer(structure)
    pyx = pyxtal()
    pyx.from_seed(structure, tol=0.01)
    space_group = pyx.group.number
    species = []
    anchors = []
    matrices = []
    coords = []
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
    sym_info = {"anchors": anchors, "wyckoff_ops": matrices, "spacegroup": space_group}
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


def get_symmetry_info(crystal, tol=0.01):
    spga = SpacegroupAnalyzer(crystal, symprec=tol)
    crystal = spga.get_refined_structure()
    c = pyxtal()
    try:
        c.from_seed(crystal, tol=0.01)
    except:
        c.from_seed(crystal, tol=0.0001)
    space_group = c.group.number
    species = []
    anchors = []
    matrices = []
    coords = []
    for site in c.atom_sites:
        specie = site.specie
        anchor = len(matrices)
        coord = site.position
        for syms in site.wp:
            species.append(specie)
            matrices.append(syms.affine_matrix)
            coords.append(syms.operate(coord))
            anchors.append(anchor)
    anchors = np.array(anchors)
    matrices = np.array(matrices)
    coords = np.array(coords) % 1.0
    sym_info = {"anchors": anchors, "wyckoff_ops": matrices, "spacegroup": space_group}
    crystal = Structure(
        lattice=Lattice.from_parameters(*np.array(c.lattice.get_para(degree=True))),
        species=species,
        coords=coords,
        coords_are_cartesian=False,
    )
    return crystal, sym_info


def eval_spg_match_rate(structures, spg, tol=0.01):
    res = {}
    for i, structure in enumerate(structures):
        sgn = SpacegroupAnalyzer(structure, symprec=0.1).get_space_group_number()
        # print(i,sgn)
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
    return res


def unzip_cif_zips(file_path: str):
    if not os.path.exists(file_path + "/generated_crystals_cif.zip"):
        print("No zip file found")
    elif os.path.exists(file_path + "/generated_crystals_cif"):
        print("Directory already exists")
    else:
        zipfile.ZipFile(file_path + "/generated_crystals_cif.zip", "r").extractall(file_path)
        os.rename(file_path + "/tmp", file_path + "/generated_crystals_cif")
    cif_path = file_path + "/generated_crystals_cif"
    print(cif_path)
    return cif_path


def get_cif_list(cif_path: str):
    cif_list = [f for f in os.listdir(cif_path) if f.endswith(".cif")]
    cif_list.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))
    cif_list = [os.path.join(cif_path, f) for f in cif_list]
    return cif_list


def refine_struc(struc):
    # Conventional cell
    spga = SpacegroupAnalyzer(struc)
    struc = spga.get_refined_structure()
    return struc


def process_cif_to_conventional(structure_cif, tol=0.01):
    structure = CifParser.from_string(structure_cif).get_structures()[0]
    # sga = SpacegroupAnalyzer(structure)
    pyx = pyxtal()
    pyx.from_seed(structure, tol=tol)
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
    sym_info = {"anchors": anchors, "wyckoff_ops": matrices, "spacegroup": space_group}
    cif = write_cif(pyx)[805:]
    num_sites = len(species)
    formula = pyx.formula
    return cif, sym_info, num_sites, formula


def process_conventional_with_sym(structure, tol=0.01):
    sga = SpacegroupAnalyzer(structure)
    pyx = pyxtal()
    pyx.from_seed(structure, tol=tol)
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
        "anchors": anchors,  # equivalent_atoms
        "wyckoff_ops": matrices,
        "spacegroup": space_group,
        "conv_to_prim": conv_to_prim.tolist(),
        "prim_to_conv": prim_to_conv.tolist(),
    }
    cif = write_cif(pyx)[805:]
    num_sites = len(species)
    formula = pyx.formula
    return cif, sym_info, num_sites, formula


def initialize_random_atoms(num_atoms):
    coords = random.random((num_atoms, 3))
    return coords


def project_sites(randoms_sites, sym_info):
    anchors = sym_info["anchors"]
    wyckoff_ops = sym_info["wyckoff_ops"]
    pos_with_trans = np.concatenate(
        [randoms_sites[anchors], np.ones((randoms_sites.shape[0], 1))], axis=1
    )
    pos_proj = np.einsum("bij, bj -> bi", np.array(wyckoff_ops), pos_with_trans)
    pos_proj = pos_proj[:, :3] % 1

    return pos_proj


def initialize_random_lattice():
    a, b, c = random.random(3) * 5 + 5
    alpha, beta, gamma = random.random(3) * 60 + 60
    return Lattice.from_parameters(a, b, c, alpha, beta, gamma)


def project_lattice(lattice, spacegroup):
    if isinstance(lattice, Lattice):
        mat = torch.Tensor(lattice.matrix.copy()).unsqueeze(0)
    elif isinstance(lattice, torch.Tensor):
        assert lattice.shape == (1, 3, 3) or lattice.shape == (3, 3)
        if lattice.shape == (3, 3):
            lattice = lattice.unsqueeze(0)
        mat = lattice
    elif isinstance(lattice, np.ndarray):
        assert lattice.shape == (1, 3, 3) or lattice.shape == (3, 3)
        if lattice.shape == (3, 3):
            lattice = lattice[np.newaxis, ...]
        mat = torch.Tensor(lattice.copy())

    cf = symmetry_utils.CrystalFamily()
    mat2 = cf.de_so3(mat)
    vec = cf.m2v(mat2)
    proj_vec = cf.proj_k_to_spacegroup(vec, spacegroup)
    proj_mat = cf.v2m(proj_vec)
    lattice = Lattice(proj_mat[0].numpy())
    return lattice


def test_space_group_projection(structure, print_res=False):
    cif, sym_info, num_sites, formula = process_conventional_with_sym(structure)
    plat = project_lattice(initialize_random_lattice(), sym_info["spacegroup"])
    psites = project_sites(initialize_random_atoms(num_sites), sym_info)
    # print(psites)
    species = SpacegroupAnalyzer(structure).get_refined_structure().species

    proj_structure = Structure(plat, species, psites)

    res = {
        "proj_structure": proj_structure,
        "structure": structure,
        "sym_info": sym_info,
    }

    try:
        spga = SpacegroupAnalyzer(proj_structure)
    except:
        return {**res, "correct": False, "error": True}
    if print_res:
        print("Original space group number: ", sym_info["spacegroup"])
        print("Projected space group number: ", spga.get_space_group_number())
    correct = sym_info["spacegroup"] == spga.get_space_group_number()
    return {**res, "correct": correct, "error": False}


def project_lattice_to_primitive_wrapper(conv, sym_info):
    return conv.lattice.matrix.dot(np.array(sym_info["conv_to_prim"]))


def test_project_lattice(struct):
    sga = SpacegroupAnalyzer(struct)
    prim = sga.get_primitive_standard_structure()
    conv = sga.get_conventional_standard_structure()
    conv_to_prim = sga.get_conventional_to_primitive_transformation_matrix()
    # prim_to_conv = np.linalg.inv(conv_to_prim)
    sgn = sga.get_space_group_number()

    prim_lat = prim.lattice.matrix
    conv_lat = conv.lattice.matrix
    res1 = project_random_lattice_to_symmetry(
        conv_lat, conv_to_prim, sgn, inp_prim=False, out_prim=True
    )
    res2 = project_random_lattice_to_symmetry(
        prim_lat, conv_to_prim, sgn, inp_prim=True, out_prim=False
    )
    # print(res1)
    # print(res2)
    return (
        np.allclose(np.array(Lattice(res1).parameters), np.array(prim.lattice.parameters)),
        np.allclose(np.array(Lattice(res2).parameters), np.array(conv.lattice.parameters)),
    )
    ## Test codes
    # for i in range(100):
    # sgn = SpacegroupAnalyzer(structures[i]).get_space_group_number()
    # symbol = SpacegroupAnalyzer(structures[i]).get_space_group_symbol()
    # res = np.array(test_project_lattice(structures[i]))
    # if any(~res):
    #     print(i, res, sgn, symbol)


def project_random_lattice_to_symmetry(
    rand_lat_mat, conv_to_prim, space_group, inp_prim=True, out_prim=False
):
    if inp_prim:
        rand_conv_lat = np.linalg.inv(conv_to_prim).dot(rand_lat_mat)
    else:
        rand_conv_lat = rand_lat_mat
    proj_conv_lat = project_lattice(rand_conv_lat, space_group)

    if out_prim:
        proj_prim_lat = conv_to_prim.dot(proj_conv_lat.matrix)
        return proj_prim_lat
    else:
        return proj_conv_lat.matrix


# def project_random_prim_lattice_to_prim(lat_mat, conv_to_prim, space_group):
#     rand_conv_lat = np.linalg.inv(conv_to_prim).dot(lat_mat)
#     proj_conv_lat = project_lattice(rand_conv_lat, space_group)
#     proj_prim_lat = conv_to_prim.dot(proj_conv_lat.matrix)
#     return proj_prim_lat


def prim_coords_to_conv_coords(frac_coords, prim_lat, conv_lat, use_is_close=True):
    res = conv_lat.get_fractional_coords(prim_lat.get_cartesian_coords(frac_coords))
    if use_is_close:
        res[np.isclose(res, np.zeros_like(res))] = 0
    return res


def periodic_norm_frac(a, b):
    mirror = [(i, j, k) for i in [-1, 0, 1] for j in [-1, 0, 1] for k in [-1, 0, 1]]
    mirror = np.array(mirror).reshape(-1, 1, 3)
    a = np.array(a) % 1
    b = np.array(b) % 1
    b_mirror = b.reshape(1, -1, 3) + mirror
    dist = np.linalg.norm(
        a.reshape(1, 1, -1, 3).repeat(27, axis=0) - b_mirror.reshape(27, -1, 1, 3), axis=3
    )
    dist_min = np.min(dist, axis=0)
    dist_argmin = dist.argmin(axis=0)
    row_indices, col_indices = linear_sum_assignment(dist_min)
    return {
        "distances": dist_min[row_indices, col_indices],
        "indices": np.stack([row_indices, col_indices]),
        "target_pos": b_mirror[dist_argmin[row_indices, col_indices], col_indices],
    }


def periodic_norm_frac_fixed(frac_1, frac_2):
    frac_1 = np.array(frac_1)
    frac_2 = np.array(frac_2)
    frac_dist = np.subtract(frac_1.reshape(1, -1, 3), frac_2.reshape(-1, 1, 3))
    frac_dist2 = np.linalg.norm(frac_dist - np.round(frac_dist), axis=-1)
    row_indices, col_indices = linear_sum_assignment(frac_dist2)
    target_pos = frac_2[col_indices] - np.round(frac_dist[row_indices, col_indices])
    return {
        "distances": frac_dist2[row_indices, col_indices],
        "indices": np.stack([row_indices, col_indices]),
        "target_pos": target_pos,
    }


def pbc_diff(frac_coords1, frac_coords2, pbc=(True, True, True)):
    """Get the 'fractional distance' between two coordinates taking into
    account periodic boundary conditions.

    Args:
        frac_coords1: First set of fractional coordinates. e.g. [0.5, 0.6,
            0.7] or [[1.1, 1.2, 4.3], [0.5, 0.6, 0.7]]. It can be a single
            coord or any array of coords.
        frac_coords2: Second set of fractional coordinates.
        pbc: a tuple defining the periodic boundary conditions along the three
            axis of the lattice.

    Returns:
        Fractional distance. Each coordinate must have the property that
        abs(a) <= 0.5. Examples:
        pbc_diff([0.1, 0.1, 0.1], [0.3, 0.5, 0.9]) = [-0.2, -0.4, 0.2]
        pbc_diff([0.9, 0.1, 1.01], [0.3, 0.5, 0.9]) = [-0.4, -0.4, 0.11]
    """
    frac_dist = np.subtract(frac_coords1, frac_coords2)
    return frac_dist - np.round(frac_dist) * pbc


def is_periodic_image(frac_coords1, frac_coords2, pbc=(True, True, True), tolerance=1e-7):
    frac_diff = pbc_diff(frac_coords1, frac_coords2, pbc)
    return np.allclose(frac_diff, np.zeros_like(frac_diff), atol=tolerance)


def map_sites_conv_to_prim(conv, prim):
    prim_lattice_inv = prim.lattice.inv_matrix
    new_fracs = []
    new_species = []
    for site in conv:
        # site.coords is in cartesian
        new_frac = np.dot(site.coords, prim_lattice_inv)
        # if not any(partial(pbc_diff,new_frac) == partial(f) for f in new_fracs):
        #     new_species.append(site.specie)
        if not any(map(partial(is_periodic_image, new_frac), new_fracs)):
            new_fracs.append(new_frac % 1)
            new_species.append(site.specie)
        new_fracs = [new_frac % 1 for new_frac in new_fracs]

    # Rhombohedral is not implemented yet
    return new_species, new_fracs


def get_prim_and_conv(structure):
    sga = SpacegroupAnalyzer(structure)
    prim = sga.get_primitive_standard_structure()
    conv = sga.get_conventional_standard_structure()
    return prim, conv


def process_prim_with_sym_infos(structure, tol=0.01):
    sga = SpacegroupAnalyzer(structure)
    pyx = pyxtal()
    pyx.from_seed(structure, tol=tol)
    space_group = pyx.group.number
    sd = sga.get_symmetry_dataset()
    space_group = sd.number
    equivalent_atoms = sd.equivalent_atoms
    crystal_system = sga.get_crystal_system()
    spacegroup_symbol = sga.get_space_group_symbol()
    lattice_type = sga.get_lattice_type()
    wyckoffs = sd.wyckoffs

    anchors = []
    matrices = []
    matrices_batch = []
    new_wyckoffs = []
    species = []
    unique = np.unique(equivalent_atoms)
    for i, u in enumerate(unique):
        wp = Wyckoff_position.from_group_and_letter(space_group, wyckoffs[u])
        anchor = len(anchors)
        for j in np.where(equivalent_atoms == u)[0]:
            anchors.append(anchor)
            new_wyckoffs.append(wyckoffs[u])
            species.append(structure.species[u].number)
        for syms in wp:
            matrices.append(syms.affine_matrix)
            matrices_batch.append(anchor)
    matrices = np.array(matrices).tolist()
    conv_to_prim = sga.get_conventional_to_primitive_transformation_matrix()
    prim_to_conv = np.linalg.inv(conv_to_prim)
    num_sites = len(anchors)
    sym_info = {
        "anchors": anchors,  # equivalent_atoms
        "wyckoff_ops": matrices,
        "wyckoff_batch": matrices_batch,
        "spacegroup": space_group,
        "conv_to_prim": conv_to_prim.tolist(),
        "prim_to_conv": prim_to_conv.tolist(),
        "crystal_system": crystal_system,
        "spacegroup_symbol": spacegroup_symbol,
        "wyckoffs": new_wyckoffs,
        "num_atoms": num_sites,
        "species": species,
        "lattice_type": lattice_type,
    }
    cif = write_cif(pyx)[805:]
    formula = pyx.formula
    return cif, sym_info, num_sites, formula


def project_sites_prim(random_sites, sym_info):
    anchors = sym_info["anchors"]
    wyckoff_ops = sym_info["wyckoff_ops"]
    sites = random_sites[sym_info["wyckoff_batch"]]
    pos_with_trans = np.concatenate([sites[anchors], np.ones((sites.shape[0], 1))], axis=1)
    pos_proj = np.einsum("bij, bj -> bi", np.array(wyckoff_ops), pos_with_trans)
    pos_proj = pos_proj[:, :3] % 1

    return pos_proj


def project_sites_prim_with_species(random_sites, sym_info):
    # anchors = sym_info["anchors"]
    wyckoff_ops = sym_info["wyckoff_ops"]
    species = np.array(sym_info["species"])
    sites = random_sites[np.array(sym_info["wyckoff_batch"])]
    species = species[np.array(sym_info["wyckoff_batch"])]
    pos_with_trans = np.concatenate([sites, np.ones((sites.shape[0], 1))], axis=1)
    pos_proj = np.einsum("bij, bj -> bi", np.array(wyckoff_ops), pos_with_trans)
    pos_proj = pos_proj[:, :3] % 1

    return pos_proj, species


# def permute_conv_lat(lat_mat):
#     rank = np.argsort(np.linalg.norm(lat_mat, axis=1))
#     perm = lat_perm[rank[0], rank[1], rank[2]]
#     perm_lat_mat = np.dot(perm, lat_mat)
#     return perm_lat_mat, perm, rank


def permute_conv_lat(lat_mat, space_group):
    rank = np.argsort(
        -np.linalg.norm(lat_mat, axis=1)
    )  # lat[rank[0]] > lat[rank[1]] > lat[rank[2]] (Descending order)
    perm = lat_perm[space_group, rank[0], rank[1], rank[2]]
    perm_lat_mat = np.dot(perm, lat_mat).dot(perm.T)
    return perm_lat_mat, perm, rank


# def project_random_lattice_to_symmetry_prim_conv(rand_lat_mat, conv_to_prim, space_group):
#     rand_conv_lat = np.linalg.inv(conv_to_prim).dot(rand_lat_mat)
#     proj_conv_lat = project_lattice(rand_conv_lat, space_group)
#     proj_prim_lat = np.array(conv_to_prim).dot(proj_conv_lat.matrix)

#     return proj_prim_lat, proj_conv_lat.matrix


def project_random_lattice_to_symmetry_prim_conv_permute(rand_lat_mat, conv_to_prim, space_group):
    rand_conv_lat = np.linalg.inv(conv_to_prim).dot(rand_lat_mat)
    proj_conv_lat = project_lattice(rand_conv_lat, space_group)
    perm_conv_lat, perm, rank = permute_conv_lat(proj_conv_lat.matrix, space_group)
    # proj_prim_lat = np.array(conv_to_prim).dot(perm_conv_lat)
    proj_prim_lat = np.array(conv_to_prim).dot(
        perm_for_A2[space_group].dot(perm_conv_lat).dot(perm_for_A2[space_group].T)
    )

    return proj_prim_lat, perm_conv_lat, perm, rank


def map_sites_conv_to_prim_lat(conv_sites, conv_species, conv_lat, prim_lat):
    prim_lattice_inv = np.linalg.inv(prim_lat)
    new_fracs = []
    new_species = []
    conv_sites = np.dot(conv_sites, conv_lat)
    for cart_coords, specie in zip(conv_sites, conv_species):
        new_frac = np.dot(cart_coords, prim_lattice_inv)
        # if not any(partial(pbc_diff,new_frac) == partial(f) for f in new_fracs):
        #     new_species.append(site.specie)
        if not any(map(partial(is_periodic_image, new_frac), new_fracs)):
            new_fracs.append(new_frac % 1)
            new_species.append(specie)
        new_fracs = [new_frac % 1 for new_frac in new_fracs]

    # Rhombohedral is not implemented yet
    return new_species, new_fracs


def map_sites_conv_to_prim_lat_triu(conv_sites, conv_species, conv_lat, prim_lat):
    prim_lattice_inv = np.linalg.inv(prim_lat)
    new_fracs = []
    new_species = []
    conv_sites = np.dot(conv_sites, conv_lat)
    for cart_coords, specie in zip(conv_sites, conv_species):
        new_frac = np.dot(cart_coords, prim_lattice_inv)
        # if not any(partial(pbc_diff,new_frac) == partial(f) for f in new_fracs):
        #     new_species.append(site.specie)
        if not any(map(partial(is_periodic_image, new_frac), new_fracs)):
            new_fracs.append(new_frac % 1)
            new_species.append(specie)
        new_fracs = [new_frac % 1 for new_frac in new_fracs]

    # Rhombohedral is not implemented yet
    return new_species, new_fracs


def test_space_group_projection_prim3(structure, print_res=False):
    cif, sym_info, num_sites, formula = process_prim_with_sym_infos(structure)
    space_group = sym_info["spacegroup"]
    prim_plat, conv_plat, perm, rank = project_random_lattice_to_symmetry_prim_conv_permute(
        initialize_random_lattice().matrix, sym_info["conv_to_prim"], sym_info["spacegroup"]
    )
    # prim_plat = np.dot(perm, prim_plat)
    conv_psites, conv_species = project_sites_prim_with_species(
        initialize_random_atoms(num_sites), sym_info
    )
    # perm_conv_psites = np.dot(perm, conv_psites.T).T
    # conv_psites = perm_conv_psites

    # print(psites)
    # conv_species = SpacegroupAnalyzer(structure).get_refined_structure().species
    # print(prim_plat)
    # print("conv_plat",conv_plat)
    proj_conv_structure = Structure(conv_plat, conv_species, conv_psites)
    perm_conv_psite = np.dot(perm, conv_psites.T).T
    perm_conv_plat = perm_for_A2[space_group].dot(conv_plat).dot(perm_for_A2[space_group].T)
    # print(prim_plat)
    prim_species, prim_psites = map_sites_conv_to_prim_lat(
        perm_conv_psite, conv_species, perm_conv_plat, prim_plat
    )

    proj_prim_structure = Structure(prim_plat, prim_species, prim_psites)

    res = {
        "proj_conv_structure": proj_conv_structure,
        "proj_prim_structure": proj_prim_structure,
        "structure": structure,
        "sym_info": sym_info,
    }

    try:
        spga = SpacegroupAnalyzer(proj_conv_structure)
    except:
        return {
            **res,
            "correct": False,
            "error": True,
            "correct_species_len": len(structure.species) == len(proj_prim_structure.species),
            "correct_species_len_spga": len(structure.species) == 0,
        }

    spga_prim = spga.get_primitive_standard_structure()
    spg_species_len = len(spga_prim.species)

    if print_res:
        print("Original space group number: ", sym_info["spacegroup"])
        print("Projected space group number: ", spga.get_space_group_number())
        print(
            "Len of species: ",
            len(structure.species),
            len(proj_prim_structure.species),
            spg_species_len,
            len(structure.species) == len(proj_prim_structure.species),
            len(structure.species) == spg_species_len,
        )
        print
    correct = sym_info["spacegroup"] == spga.get_space_group_number()
    return {
        **res,
        "spga_prim": spga_prim,
        "correct": correct,
        "error": False,
        "correct_species_len": len(structure.species) == len(proj_prim_structure.species),
        "correct_species_len_spga": len(structure.species) == spg_species_len,
    }


def test_space_group_projection_prim4(structure, print_res=False):
    cif, sym_info, num_sites, formula = process_prim_with_sym_infos(structure)
    space_group = sym_info["spacegroup"]
    prim_plat, conv_plat, perm, rank = project_random_lattice_to_symmetry_prim_conv_permute(
        initialize_random_lattice().matrix, sym_info["conv_to_prim"], sym_info["spacegroup"]
    )
    # prim_plat = np.dot(perm, prim_plat)
    conv_psites, conv_species = project_sites_prim_with_species(
        initialize_random_atoms(num_sites), sym_info
    )
    # perm_conv_psites = np.dot(perm, conv_psites.T).T
    # conv_psites = perm_conv_psites

    # print(psites)
    # conv_species = SpacegroupAnalyzer(structure).get_refined_structure().species
    # print(prim_plat)
    # print("conv_plat",conv_plat)
    proj_conv_structure = Structure(conv_plat, conv_species, conv_psites)
    perm_conv_psite = np.dot(perm, conv_psites.T).T
    perm_conv_plat = perm_for_A2[space_group].dot(conv_plat).dot(perm_for_A2[space_group].T)
    # print(prim_plat)
    prim_species, prim_psites = map_sites_conv_to_prim_lat(
        perm_conv_psite, conv_species, perm_conv_plat, prim_plat
    )

    proj_prim_structure = Structure(prim_plat, prim_species, prim_psites)

    res = {
        "proj_conv_structure": proj_conv_structure,
        "proj_prim_structure": proj_prim_structure,
        "structure": structure,
        "sym_info": sym_info,
    }

    try:
        spga = SpacegroupAnalyzer(proj_conv_structure)
    except:
        return {
            **res,
            "correct": False,
            "error": True,
            "correct_species_len": len(structure.species) == len(proj_prim_structure.species),
            "correct_species_len_spga": len(structure.species) == 0,
        }

    spga_prim = spga.get_primitive_standard_structure()
    spg_species_len = len(spga_prim.species)

    if print_res:
        print("Original space group number: ", sym_info["spacegroup"])
        print("Projected space group number: ", spga.get_space_group_number())
        print(
            "Len of species: ",
            len(structure.species),
            len(proj_prim_structure.species),
            spg_species_len,
            len(structure.species) == len(proj_prim_structure.species),
            len(structure.species) == spg_species_len,
        )
        print
    correct = sym_info["spacegroup"] == spga.get_space_group_number()
    return {
        **res,
        "spga_prim": spga_prim,
        "correct": correct,
        "error": False,
        "correct_species_len": len(structure.species) == len(proj_prim_structure.species),
        "correct_species_len_spga": len(structure.species) == spg_species_len,
    }


# def test_space_group_projection_prim(structure, print_res=False):
#     cif, sym_info, num_sites, formula = process_prim_with_sym_infos(structure)
#     prim_plat, conv_plat = project_random_lattice_to_symmetry_prim_conv(
#         initialize_random_lattice().matrix, sym_info["conv_to_prim"], sym_info["spacegroup"]
#     )
#     conv_psites, conv_species = project_sites_prim_with_species(
#         initialize_random_atoms(num_sites), sym_info
#     )
#     # print(psites)
#     # conv_species = SpacegroupAnalyzer(structure).get_refined_structure().species

#     proj_conv_structure = Structure(conv_plat, conv_species, conv_psites)
#     prim_species, prim_psites = map_sites_conv_to_prim_lat(
#         conv_psites, conv_species, conv_plat, prim_plat
#     )
#     proj_prim_structure = Structure(prim_plat, prim_species, prim_psites)

#     res = {
#         "proj_conv_structure": proj_conv_structure,
#         "proj_prim_structure": proj_prim_structure,
#         "structure": structure,
#         "sym_info": sym_info,
#     }

#     try:
#         spga = SpacegroupAnalyzer(proj_conv_structure)
#     except:
#         return {
#             **res,
#             "correct": False,
#             "error": True,
#             "correct_species_len": len(structure.species) == len(proj_prim_structure.species),
#             "correct_species_len_spga": len(structure.species) == 0,
#         }

#     spga_prim = spga.get_primitive_standard_structure()
#     spg_species_len = len(spga_prim.species)

#     if print_res:
#         print("Original space group number: ", sym_info["spacegroup"])
#         print("Projected space group number: ", spga.get_space_group_number())
#         print(
#             "Len of species: ",
#             len(structure.species),
#             len(proj_prim_structure.species),
#             spg_species_len,
#             len(structure.species) == len(proj_prim_structure.species),
#             len(structure.species) == spg_species_len,
#         )
#         print
#     correct = sym_info["spacegroup"] == spga.get_space_group_number()
#     return {
#         **res,
#         "spga_prim": spga_prim,
#         "correct": correct,
#         "error": False,
#         "correct_species_len": len(structure.species) == len(proj_prim_structure.species),
#         "correct_species_len_spga": len(structure.species) == spg_species_len,
#     }


# Spg_data
# ['number',
#  'hall_number',
#  'international',
#  'hall',
#  'choice',
#  'transformation_matrix',
#  'origin_shift',
#  'rotations',
#  'translations',
#  'wyckoffs',
#  'site_symmetry_symbols',
#  'crystallographic_orbits',
#  'equivalent_atoms',
#  'primitive_lattice',
#  'mapping_to_primitive',
#  'std_lattice',
#  'std_positions',
#  'std_types',
#  'std_rotation_matrix',
#  'std_mapping_to_primitive',
#  'pointgroup']
