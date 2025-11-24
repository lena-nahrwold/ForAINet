import os
import torch
import numpy as np
from plyfile import PlyData
from torch_geometric.data import Data, Dataset as TorchDataset
from torch_points3d.datasets.base_dataset import BaseDataset


# -------------------------------------------------------------------------
# Empty dataset (TorchPoints3D expects train/val to exist)
# -------------------------------------------------------------------------
class EmptyDataset(TorchDataset):
    def __init__(self):
        super().__init__("")

    def len(self):
        return 0

    def get(self, idx):
        return None


# -------------------------------------------------------------------------
# Single-file PLY loader for inference
# -------------------------------------------------------------------------
class SinglePLYDataset(TorchDataset):
    def __init__(self, root, num_classes=5):
        super().__init__(root)
        self.num_classes_value = num_classes
        self.files = [f for f in os.listdir(root) if f.endswith(".ply")]

    def len(self):
        return len(self.files)

    def get(self, idx):
        filepath = os.path.join(self.root, self.files[idx])
        plydata = PlyData.read(filepath)
        vertices = plydata["vertex"]

        pts = np.vstack([vertices["x"], vertices["y"], vertices["z"]]).T
        pts = torch.tensor(pts, dtype=torch.float) # [N, 3]

        fake_feat = torch.ones((pts.shape[0], 1), dtype=torch.float)  # [N, 1]
        feats = torch.cat([pts, fake_feat], dim=1)  # [N, 4]
        data = Data(pos=pts, x=feats)

        # -----------------------------
        # REQUIRED FOR PANOPTIC MODELS
        # -----------------------------
        data.y = torch.zeros(pts.shape[0], dtype=torch.long)               # dummy semantic labels
        data.instance_labels = torch.zeros(pts.shape[0], dtype=torch.long) # dummy instance IDs
        data.num_classes = self.num_classes_value                          # required by BaseDataset.num_classes

        return data


# -------------------------------------------------------------------------
# TorchPoints3D dataset wrapper
# -------------------------------------------------------------------------
class PLYInferenceDataset(BaseDataset):
    def __init__(self, dataset_opt):
        super().__init__(dataset_opt)

        root = dataset_opt.dataroot

        # TorchPoints3D expects dataset split:
        self.train_dataset = EmptyDataset()
        self.val_dataset = EmptyDataset()
        self.test_dataset = SinglePLYDataset(root)

        # Metadata required by PointGroup3heads
        self._num_classes = 5

        self._label_to_names = {
            0: "ground",
            1: "tree",
            2: "fallen_tree",
            3: "sapling",
            4: "shrub",
        }

        self._stuff_classes = [0]      # usually ground
        self._thing_classes = [1, 2, 3, 4]

    # TorchPoints3D expects these properties:
    @property
    def num_classes(self):
        return self._num_classes

    @property
    def stuff_classes(self):
        return self._stuff_classes

    @property
    def thing_classes(self):
        return self._thing_classes

    @property
    def label_to_names(self):
        return self._label_to_names