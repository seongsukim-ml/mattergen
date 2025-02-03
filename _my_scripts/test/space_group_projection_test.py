import sys
import os
from pathlib import Path

sys.path.append(Path.cwd().parent.as_posix())
from util import *
from visualizer import visualizer

vis = visualizer()

if __name__ == "__main__":
    ncpu = 16

    structures = test_structures(N=-1, dataset_name="alex_mp_20", dataset_mode="val", num_cpus=ncpu)

    eval = p_map(test_space_group_projection, structures, num_cpus=ncpu)
    print("Number of failed: ", len([i for i in eval if i["correct"] == False]))
    print("Success rate: ", len([i for i in eval if i["correct"] == True]) / len(eval))
    print("Ratio of correct space group: ", sum([i["correct"] for i in eval]), len(eval))

## alex_mp_20_train
# Number of failed:  2894
# Success rate:      0.99523
# Ratio of correct space group: 604789 / 607683
