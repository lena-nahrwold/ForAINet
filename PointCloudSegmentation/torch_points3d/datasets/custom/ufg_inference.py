import os
import torch
import numpy as np
from torch_geometric.io import read_ply
from torch_points3d.datasets.base_dataset import BaseDataset
from torch_points3d.metrics.segmentation_tracker import SegmentationTracker
from torch_geometric.data import InMemoryDataset


class PlyInference(InMemoryDataset):
    def __init__(self, root, ply_path, num_classes=5, transform=None, pre_transform=None, pre_filter=None):
        self.ply_path = ply_path
        self.num_classes_value = num_classes
        super().__init__(root, transform, pre_transform, pre_filter)
        self.load(self.processed_paths[0])
        # For PyG<2.4:
        # self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [os.path.basename(self.ply_path)]

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass

    def process(self):
        # Read the PLY file
        data = read_ply(self.ply_path)
        data.pos = data.pos.float()  # convert to float

        N = data.pos.shape[0]

        data.x = torch.cat([data.pos, torch.ones((data.pos.shape[0], 1))], dim=1)
        data.y = torch.zeros(N, dtype=torch.long)               # dummy semantic labels
        data.instance_labels = torch.zeros(N, dtype=torch.long) # dummy instance IDs
        data.center_label = torch.zeros(data.pos.shape[0], dtype=torch.long)
        data.num_classes = self.num_classes_value   
        data.num_instances = 100
        data.instance_mask = torch.zeros(data.pos.shape[0], dtype=torch.bool)
        data.vote_label = torch.zeros(data.pos.shape[0], 3, dtype=torch.float)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        # Wrap it in a list, as PyG expects a list of Data objects
        data_list = [data]

        # Save in-memory dataset
        self.save(data_list, self.processed_paths[0])


class PlyInferenceDataset(BaseDataset):
    def __init__(self, dataset_opt):
        super().__init__(dataset_opt)

        ply_path = os.path.join(self._data_path, dataset_opt.ply_file)

        self.train_dataset = PlyInference(
            root=self._data_path,
            ply_path=ply_path,
            pre_transform=None,
            transform=None,
        )

        self.test_dataset = PlyInference(
            root=self._data_path,
            ply_path=ply_path,
            pre_transform=self.pre_collate_transform,
            transform=self.test_transform,
        )

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

    def get_tracker(self, wandb_log: bool, tensorboard_log: bool):
        """Factory method for the tracker

        Arguments:
            wandb_log - Log using weight and biases
            tensorboard_log - Log using tensorboard
        Returns:
            [BaseTracker] -- tracker
        """
        return SegmentationTracker(self, wandb_log=wandb_log, use_tensorboard=tensorboard_log)

