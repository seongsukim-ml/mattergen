from mattergen.common.data.dataset import *
from mattergen.common.data.condition_factory import *
from mattergen.common.data.condition_factory import _collate_fn
import json


def get_wyckoff_condition_loader(
    # space_group_counts: dict[int, int],
    space_group_infos_path: str,
    num_samples: int,
    batch_size: int,
    shuffle: bool = False,
    transforms: list[Transform] | None = None,
    space_groups: list[int] | None = None,
    properties: TargetProperty | None = None,
) -> ConditionLoader:
    transforms = transforms or []
    if properties is not None:
        for k, v in properties.items():
            transforms.append(SetProperty(k, v))
    dataset = WyckoffDataset.from_space_group_list(
        # space_group_counts=space_group_counts,
        space_group_infos_path=space_group_infos_path,
        num_samples=num_samples,
        transforms=transforms,
        space_groups=space_groups,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=partial(_collate_fn, collate_fn=collate),
        shuffle=shuffle,
    )


@dataclass(frozen=True, kw_only=True)
class WyckoffDataset(BaseDataset):

    anchors: list[list[int]]
    uniques: list[int]
    space_groups: numpy.typing.NDArray
    wyckoff_ops: list[numpy.typing.NDArray]
    wyckoff_batch: list[numpy.typing.NDArray]
    conv_to_prim: list[numpy.typing.NDArray]
    prim_to_conv: list[numpy.typing.NDArray]
    species: list[numpy.typing.NDArray]
    num_atoms: numpy.typing.NDArray | None
    structure_id: numpy.typing.NDArray | None = None
    properties: dict[PropertySourceId, numpy.typing.NDArray] = field(default_factory=dict)
    transforms: list[Transform] | None = None

    def __getitem__(self, index: int) -> ChemGraph:
        num_atoms = torch.tensor(self.num_atoms[index], dtype=torch.long)
        anchors = torch.tensor(self.anchors[index], dtype=torch.long)
        wyckoff_ops = torch.tensor(self.wyckoff_ops[index], dtype=torch.float)
        wyckoff_ops_pinv = torch.linalg.pinv(wyckoff_ops[:, :3, :3])
        wyckoff_batch = torch.tensor(self.wyckoff_batch[index], dtype=torch.long)
        space_group = torch.tensor(self.space_groups[index], dtype=torch.long)
        conv_to_prim = torch.tensor(self.conv_to_prim[index], dtype=torch.float).reshape(1, 3, 3)
        prim_to_conv = torch.tensor(self.prim_to_conv[index], dtype=torch.float).reshape(1, 3, 3)
        species = torch.tensor(self.species[index], dtype=torch.long)
        uniques = torch.tensor(self.uniques[index], dtype=torch.long)

        props_dict = self.get_properties_dict(index)

        data = ChemGraph(
            pos=torch.full((num_atoms, 3), fill_value=torch.nan, dtype=torch.float),
            cell=torch.full((1, 3, 3), fill_value=torch.nan, dtype=torch.float),
            atomic_numbers=torch.full((num_atoms,), fill_value=-1, dtype=torch.long),
            anchors=anchors,
            anchors_len=len(anchors),
            wyckoff_ops=wyckoff_ops,
            wyckoff_ops_pinv=wyckoff_ops_pinv,
            wyckoff_bat=wyckoff_batch,
            wyckoff_bat_len=len(wyckoff_batch),
            space_groups=space_group,
            num_atoms=num_atoms,
            conv_to_prim=conv_to_prim,
            prim_to_conv=prim_to_conv,
            species=species,
            uniques=uniques,
            uniques_len=len(uniques),
            num_nodes=num_atoms,  # special attribute used for batching in pytorch geometric
            **props_dict,  # type: ignore
        )
        return data

    def __len__(self) -> int:
        return len(self.num_atoms)

    def subset(self, indices: Sequence[int]) -> "WyckoffDataset":
        return WyckoffDataset(
            num_atoms=self.num_atoms[indices],
            anchors=[self.anchors[i] for i in indices],
            wyckoff_ops=self.wyckoff_ops[indices],
            space_groups=self.space_groups[indices],
            structure_id=self.structure_id[indices] if self.structure_id is not None else None,
            properties={k: v[indices] for k, v in self.properties.items()},
            transforms=self.transforms,
        )

    def repeat(self, repeats: int) -> "WyckoffDataset":
        num_atoms = repeat_along_first_axis(self.num_atoms, repeats)
        structure_id = (
            repeat_along_first_axis(self.structure_id, repeats)
            if self.structure_id is not None
            else None
        )
        properties = {k: repeat_along_first_axis(v, repeats) for k, v in self.properties.items()}
        anchors = [self.anchors[i] for i in range(len(self.anchors)) for _ in range(repeats)]
        wyckoff_ops = repeat_along_first_axis(self.wyckoff_ops, repeats)
        space_groups = repeat_along_first_axis(self.space_groups, repeats)
        return WyckoffDataset(
            num_atoms=num_atoms,
            anchors=anchors,
            wyckoff_ops=wyckoff_ops,
            space_groups=space_groups,
            structure_id=structure_id,
            properties=properties,
            transforms=self.transforms,
        )

    @classmethod
    def load_space_group_infos(cls: Type[T], space_group_infos_path: str) -> dict[int, list]:
        with open(space_group_infos_path + "/sg_infos.json", "r") as f:
            return json.load(f)

    @classmethod
    def load_space_group_counts(cls: Type[T], space_group_infos_path: str) -> dict[int, int]:
        with open(space_group_infos_path + "/sg_counts.json", "r") as f:
            return json.load(f)

    @classmethod
    def from_space_group_list(
        cls: Type[T],
        space_group_infos_path: str,
        # space_group_counts: dict[int, int],
        # space_group_infos: dict[int, list],
        num_samples: int,
        transforms: list[Transform] | None = None,
        space_groups: list[int] | int | None = None,
    ) -> T:
        """
        space_group_infos_path: str, path to the directory containing the space group infos
        num_samples: int, number of samples to generate
        transforms: list[Transform] | None, list of transforms to apply to the dataset
        space_groups: list[int] | None, list of space groups condition to sample from. if None, samples are drawn from space groups distribution

        """
        space_group_counts = cls.load_space_group_counts(space_group_infos_path)
        space_group_infos = cls.load_space_group_infos(space_group_infos_path)

        if space_groups is None:
            total_count = sum(space_group_counts.values())
            space_group_probs = [count / total_count for count in space_group_counts.values()]
            space_groups = np.random.choice(
                list(space_group_counts.keys()), num_samples, p=space_group_probs
            )
        else:
            if isinstance(space_groups, int):
                space_groups = [space_groups for _ in range(num_samples)]
            elif len(space_groups) == 1:
                space_groups = [space_groups[0] for _ in range(num_samples)]
            else:
                assert len(space_group_counts) == len(
                    space_groups
                ), "space_groups must have the same length as space_group_counts"

        infos = []

        for i, space_group in enumerate(space_groups):
            rand = np.random.randint(0, space_group_counts[str(space_group)])
            infos.append(space_group_infos[str(space_group)][rand])

        num_atoms = np.zeros(num_samples, dtype=int)
        anchors = []
        species = []
        uniques = []
        wyckoff_ops = []
        wyckoff_batch = []
        structure_id = []
        conv_to_prim = []
        prim_to_conv = []

        for i, info in enumerate(infos):
            # num_atoms[i] = info["num_sites"]
            # num_atoms[i] = len(info["anchors"])
            num_atoms[i] = info["num_atoms"]
            anchors.append(info["anchors"])
            # anchors.append(info["wyckoff_batch"])
            uniques.append(info["anchors"])
            species.append(np.array(info["species"]))
            wyckoff_ops.append(np.array(info["wyckoff_ops"]))
            wyckoff_batch.append(np.array(info["wyckoff_batch"]))
            conv_to_prim.append(np.array(info["conv_to_prim"]))
            prim_to_conv.append(np.array(info["prim_to_conv"]))
            structure_id.append(info["key"])

        return WyckoffDataset(
            num_atoms=num_atoms,
            anchors=anchors,
            wyckoff_ops=wyckoff_ops,
            space_groups=np.array(space_groups),
            wyckoff_batch=wyckoff_batch,
            conv_to_prim=conv_to_prim,
            prim_to_conv=prim_to_conv,
            species=species,
            uniques=uniques,
            transforms=transforms,
            structure_id=structure_id,
        )
