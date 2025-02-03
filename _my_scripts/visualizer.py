import matplotlib.pyplot as plt
from ase.visualize.plot import plot_atoms
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.structure_matcher import StructureMatcher


class visualizer:

    def __init__(self, RADII=0.5, ROTATION="30x,30y,0z", primitive_cell=False):
        self.RADII = RADII
        self.ROTATION = ROTATION
        self.matcher = StructureMatcher(primitive_cell=primitive_cell)
        self.bridge = AseAtomsAdaptor()

    def plot_atoms(self, structure, figsize=(5, 5)):
        atoms = self.bridge.get_atoms(structure)
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        plot_atoms(atoms, ax, radii=self.RADII, rotation=self.ROTATION)
