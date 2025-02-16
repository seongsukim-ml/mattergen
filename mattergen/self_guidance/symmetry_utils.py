# From DiffCSP ++

import torch
import torch.nn as nn
import numpy as np
import math
import torch.linalg as linalg

# abs_cap


from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pyxtal import pyxtal
from torch_scatter import scatter_max, scatter_min

EPSILON = 1e-5


def get_latttice_permutations(device="cpu"):
    lat_perm = np.zeros((231, 3, 3, 3, 3, 3))
    lat_perm[:, :, :, :, 0, 0] = 1
    lat_perm[:, :, :, :, 1, 1] = 1
    lat_perm[:, :, :, :, 2, 2] = 1
    # Ex) 1, 0, 2 implies lat(1) > lat(0) > lat(2)
    possible_perms = [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]]

    perm_for_A1 = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
    perm_for_A2 = np.eye(3).reshape(1, 3, 3).repeat(231, axis=0)

    # orthorhombic with A:
    for i in [38, 39, 40, 41]:
        for p in possible_perms:
            lat_perm[i, p[0], p[1], p[2]] = perm_for_A1 @ lat_perm[0, p[0], p[1], p[2]]
        perm_for_A2[i] = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
    lat_perm = torch.tensor(lat_perm, dtype=torch.float32)
    perm_for_A1 = torch.tensor(perm_for_A1, dtype=torch.float32)
    perm_for_A2 = torch.tensor(perm_for_A2, dtype=torch.float32)
    return lat_perm.to(device), perm_for_A1.to(device), perm_for_A2.to(device)


def get_latttice_permutations2(device="cpu"):
    lat_perm = np.zeros((231, 3, 3, 3, 3, 3))
    lat_perm[:, :, :, :, 0, 0] = 1
    lat_perm[:, :, :, :, 1, 1] = 1
    lat_perm[:, :, :, :, 2, 2] = 1
    # Ex) 1, 0, 2 implies lat(1) > lat(0) > lat(2)
    possible_perms = [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]]

    perm_for_A1 = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
    perm_for_A2 = lat_perm.copy()
    # lat_perm[:, 0, 1, 2] = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
    # lat_perm[:, 0, 2, 1] = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
    # lat_perm[:, 1, 0, 2] = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    # lat_perm[:, 1, 2, 0] = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
    # lat_perm[:, 2, 0, 1] = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
    # lat_perm[:, 2, 1, 0] = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    # orthorhombic with A:
    for i in [38, 39, 40, 41]:
        perm_for_A2[i, 0, 1, 2] = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
        perm_for_A2[i, 0, 2, 1] = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
        perm_for_A2[i, 1, 0, 2] = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
        perm_for_A2[i, 1, 2, 0] = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
        perm_for_A2[i, 2, 0, 1] = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
        perm_for_A2[i, 2, 1, 0] = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])

        # perm_for_A2[i, 0, 1, 2] = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
        # perm_for_A2[i, 0, 2, 1] = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        # perm_for_A2[i, 1, 0, 2] = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
        # perm_for_A2[i, 1, 2, 0] = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
        # perm_for_A2[i, 2, 0, 1] = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
        # perm_for_A2[i, 2, 1, 0] = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

        lat_perm[i, 0, 1, 2] = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
        lat_perm[i, 0, 2, 1] = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
        lat_perm[i, 1, 0, 2] = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
        lat_perm[i, 1, 2, 0] = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
        lat_perm[i, 2, 0, 1] = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
        lat_perm[i, 2, 1, 0] = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
        # for p in possible_perms:
        #     lat_perm[i, p[0], p[1], p[2]] = perm_for_A1 @ lat_perm[0, p[0], p[1], p[2]]
        # perm_for_A2[i] = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
    lat_perm = torch.tensor(lat_perm, dtype=torch.float32)
    perm_for_A1 = torch.tensor(perm_for_A1, dtype=torch.float32)
    perm_for_A2 = torch.tensor(perm_for_A2, dtype=torch.float32)
    return lat_perm.to(device), perm_for_A1.to(device), perm_for_A2.to(device)


# def get_latttice_permutations(device="cpu"):
#     lat_perm = np.zeros((231, 3, 3, 3, 3, 3))
#     lat_perm[:, :, :, :, 0, 0] = 1
#     lat_perm[:, :, :, :, 1, 1] = 1
#     lat_perm[:, :, :, :, 2, 2] = 1
#     # 1, 0, 2 implies lat(1) > lat(0) > lat(2)
#     possible_perms = [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]]

#     perm_for_A1 = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
#     perm_for_A2 = np.eye(3).reshape(1, 3, 3).repeat(231, axis=0)

#     # orthorhombic with A:
#     for i in [38, 39, 40, 41]:
#         lat_perm[i, 0, 1, 2] = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
#         lat_perm[i, 0, 2, 1] = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
#         lat_perm[i, 1, 0, 2] = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
#         lat_perm[i, 1, 2, 0] = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
#         lat_perm[i, 2, 0, 1] = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
#         lat_perm[i, 2, 1, 0] = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

#         for p in possible_perms:
#             lat_perm[i, p[0], p[1], p[2]] = perm_for_A1 @ lat_perm[0, p[0], p[1], p[2]]
#         perm_for_A2[i] = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
#     lat_perm = torch.tensor(lat_perm, dtype=torch.float32)
#     perm_for_A1 = torch.tensor(perm_for_A1, dtype=torch.float32)
#     perm_for_A2 = torch.tensor(perm_for_A2, dtype=torch.float32)
#     return lat_perm.to(device), perm_for_A1.to(device), perm_for_A2.to(device)


def abs_cap(val, max_abs_val=1):
    """
    Returns the value with its absolute value capped at max_abs_val.
    Particularly useful in passing values to trignometric functions where
    numerical errors may result in an argument > 1 being passed in.
    https://github.com/materialsproject/pymatgen/blob/b789d74639aa851d7e5ee427a765d9fd5a8d1079/pymatgen/util/num.py#L15
    Args:
        val (float): Input value.
        max_abs_val (float): The maximum absolute value for val. Defaults to 1.
    Returns:
        val if abs(val) < 1 else sign of val * max_abs_val.
    """
    return max(min(val, max_abs_val), -max_abs_val)


def build_crystal(crystal_str, niggli=True, primitive=False):
    """Build crystal from cif string."""
    crystal = Structure.from_str(crystal_str, fmt="cif")

    if primitive:
        crystal = crystal.get_primitive_structure()

    if niggli:
        crystal = crystal.get_reduced_structure()

    canonical_crystal = Structure(
        lattice=Lattice.from_parameters(*crystal.lattice.parameters),
        species=crystal.species,
        coords=crystal.frac_coords,
        coords_are_cartesian=False,
    )
    # match is gaurantteed because cif only uses lattice params & frac_coords
    # assert canonical_crystal.matches(crystal)
    return canonical_crystal


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


def lattice_params_to_matrix_torch(lengths, angles):
    """Batched torch version to compute lattice matrix from params.

    lengths: torch.Tensor of shape (N, 3), unit A
    angles: torch.Tensor of shape (N, 3), unit degree
    """
    angles_r = torch.deg2rad(angles)
    coses = torch.cos(angles_r)
    sins = torch.sin(angles_r)

    val = (coses[:, 0] * coses[:, 1] - coses[:, 2]) / (sins[:, 0] * sins[:, 1])
    # Sometimes rounding errors result in values slightly > 1.
    val = torch.clamp(val, -1.0, 1.0)
    gamma_star = torch.arccos(val)

    vector_a = torch.stack(
        [
            lengths[:, 0] * sins[:, 1],
            torch.zeros(lengths.size(0), device=lengths.device),
            lengths[:, 0] * coses[:, 1],
        ],
        dim=1,
    )
    vector_b = torch.stack(
        [
            -lengths[:, 1] * sins[:, 0] * torch.cos(gamma_star),
            lengths[:, 1] * sins[:, 0] * torch.sin(gamma_star),
            lengths[:, 1] * coses[:, 0],
        ],
        dim=1,
    )
    vector_c = torch.stack(
        [
            torch.zeros(lengths.size(0), device=lengths.device),
            torch.zeros(lengths.size(0), device=lengths.device),
            lengths[:, 2],
        ],
        dim=1,
    )

    return torch.stack([vector_a, vector_b, vector_c], dim=1)


def lattice_matrix_to_params(matrix):
    lengths = np.sqrt(np.sum(matrix**2, axis=1)).tolist()

    angles = np.zeros(3)
    for i in range(3):
        j = (i + 1) % 3
        k = (i + 2) % 3
        angles[i] = abs_cap(np.dot(matrix[j], matrix[k]) / (lengths[j] * lengths[k]))
    angles = np.arccos(angles) * 180.0 / np.pi
    a, b, c = lengths
    alpha, beta, gamma = angles
    return a, b, c, alpha, beta, gamma


def lattices_to_params_shape(lattices):

    lengths = torch.sqrt(torch.sum(lattices**2, dim=-1))
    angles = torch.zeros_like(lengths)
    for i in range(3):
        j = (i + 1) % 3
        k = (i + 2) % 3
        angles[..., i] = torch.clamp(
            torch.sum(lattices[..., j, :] * lattices[..., k, :], dim=-1)
            / (lengths[..., j] * lengths[..., k]),
            -1.0,
            1.0,
        )
    angles = torch.arccos(angles) * 180.0 / np.pi

    return lengths, angles


def logm(A):
    det = torch.det(A)
    mask = ~(det > 0)
    b = mask.sum()
    A[mask] = torch.eye(3).unsqueeze(0).to(A).expand(b, -1, -1)
    eigenvalues, eigenvectors = linalg.eig(A)
    eigenvalues_log = eigenvalues.log()
    if torch.any(torch.isinf(eigenvalues_log)):
        mask2 = torch.isinf(eigenvalues_log).any(dim=-1)
        eigenvalues_temp, eigenvectors_temp = linalg.eig(A[mask2] + EPSILON)
        eigenvalues_log[mask2] = eigenvalues_temp.log()
        eigenvectors[mask2] = eigenvectors_temp

        if torch.any(torch.isinf(eigenvalues_log)):
            mask3 = torch.isinf(eigenvalues_log).any(dim=-1)
            A[mask3] = torch.eye(3).unsqueeze(0).to(A).expand(mask3.sum(), -1, -1)
            eigenvalues_temp, eigenvectors_temp = linalg.eig(A[mask3])
            eigenvalues_log[mask3] = eigenvalues_temp.log()
            eigenvectors[mask3] = eigenvectors_temp

    return torch.einsum(
        "bij,bj,bjk->bik", eigenvectors, eigenvalues_log, torch.linalg.inv(eigenvectors)
    ).real


def expm(A):
    return torch.matrix_exp(A)


def sqrtm(A):
    det = torch.det(A)
    mask = ~(det > 0)
    b = mask.sum()
    A[mask] = torch.eye(3).unsqueeze(0).to(A).expand(b, -1, -1)
    eigenvalues, eigenvectors = linalg.eig(A)
    return torch.einsum(
        "bij,bj,bjk->bik", eigenvectors, eigenvalues.sqrt(), torch.linalg.inv(eigenvectors)
    ).real


class CrystalFamily(nn.Module):

    def __init__(self):

        super(CrystalFamily, self).__init__()

        basis = self.get_basis()
        masks, biass = self.get_spacegroup_constraints()
        family = self.get_family_idx()

        self.register_buffer("basis", basis)
        self.register_buffer("masks", masks)
        self.register_buffer("biass", biass)
        self.register_buffer("family", family)

    def set_device(self, device):
        self.basis = self.basis.to(device)
        self.masks = self.masks.to(device)
        self.biass = self.biass.to(device)
        self.family = self.family.to(device)

    def get_basis(self):

        basis = torch.FloatTensor(
            [
                [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
                [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 0.0]],
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -2.0]],
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            ]
        )

        # Normalize
        basis = basis / basis.norm(dim=(-1, -2)).unsqueeze(-1).unsqueeze(-1)

        return basis

    def get_spacegroup_constraint(self, spacegroup):

        mask = torch.ones(6)
        bias = torch.zeros(6)

        if 195 <= spacegroup <= 230:
            pos = [0, 1, 2, 3, 4]
            mask[pos] = 0.0

        elif 143 <= spacegroup <= 194:
            pos = [0, 1, 2, 3]
            mask[pos] = 0.0
            bias[0] = (
                -0.25 * np.log(3) * np.sqrt(2)
            )  # As the basis is normalized, the coefficient here should additionally multiply the original norm of B1, that is $\sqrt{2}$

        elif 75 <= spacegroup <= 142:
            pos = [0, 1, 2, 3]
            mask[pos] = 0.0

        elif 16 <= spacegroup <= 74:
            pos = [0, 1, 2]
            mask[pos] = 0.0

        elif 3 <= spacegroup <= 15:
            pos = [0, 2]
            mask[pos] = 0.0

        elif 0 <= spacegroup <= 2:
            pass

        return mask, bias

    def get_spacegroup_constraints(self):

        masks, biass = [], []

        for i in range(231):
            mask, bias = self.get_spacegroup_constraint(i)
            masks.append(mask.unsqueeze(0))
            biass.append(bias.unsqueeze(0))

        return torch.cat(masks, dim=0), torch.cat(biass, dim=0)

    def get_family_idx(self):

        family = []
        for spacegroup in range(231):
            if 195 <= spacegroup <= 230:
                family.append(6)

            elif 143 <= spacegroup <= 194:
                family.append(5)

            elif 75 <= spacegroup <= 142:
                family.append(4)

            elif 16 <= spacegroup <= 74:
                family.append(3)

            elif 3 <= spacegroup <= 15:
                family.append(2)

            elif 0 <= spacegroup <= 2:
                family.append(1)
        return torch.LongTensor(family)

    def de_so3(self, L):

        # L: B * 3 * 3

        LLT = L @ L.transpose(-1, -2)
        L_sym = sqrtm(LLT)
        return L_sym

    def v2m(self, vec):

        batch_size, dims = vec.shape
        if dims == 6:
            basis = self.basis
        elif dims == 5:
            basis = self.basis[:-1]
        log_mat = torch.einsum("bk, kij -> bij", vec, basis)
        mat = expm(log_mat)
        return mat

    def m2v(self, mat):

        # mat: B * 3 * 3

        log_mat = logm(mat)
        # \sum_{i,k} B_ij B_jk = 1
        vec = torch.einsum("bij, kij -> bk", log_mat, self.basis)
        return vec

    def proj_k_to_spacegroup(self, vec, spacegroup):

        batch_size, dims = vec.shape
        if dims == 6:
            masks = self.masks[spacegroup, :]  # B * 6
            biass = self.biass[spacegroup, :]  # B * 6
        elif dims == 5:
            # - volume
            masks = self.masks[spacegroup, :-1]  # B * 5
            biass = self.biass[spacegroup, :-1]  # B * 5
        return vec * masks + biass
