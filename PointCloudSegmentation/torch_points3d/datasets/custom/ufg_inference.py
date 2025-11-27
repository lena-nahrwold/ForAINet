import os
import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.io import read_ply

from torch_points3d.datasets.base_dataset import BaseDataset
from torch_points3d.metrics.segmentation_tracker import SegmentationTracker

class EmptyDataset(InMemoryDataset):
    def __init__(self, root):
        super().__init__(root)
        # Save empty list
        self.save([], self.processed_paths[0])
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ["data.pt"]

    def download(self):
        pass

    def process(self):
        self.save([], self.processed_paths[0])

class PLYSingleCloudDataset(InMemoryDataset):
    def __init__(self, root, ply_path, transform=None, pre_transform=None, pre_filter=None):
        self.ply_path = ply_path
        super().__init__(root, transform, pre_transform, pre_filter)

        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [os.path.basename(self.ply_path)]

    @property
    def processed_file_names(self):
        return ["data.pt"]

    def download(self):
        pass

    def process(self):
        data = read_ply(self.ply_path)
        data.pos = data.pos.float()

        data_list = [data]

        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        self.save(data_list, self.processed_paths[0])


class PLYInferenceDataset(BaseDataset):
    def __init__(self, dataset_opt):
        super().__init__(dataset_opt)

        ply_path = os.path.join(self._data_path, dataset_opt.ply_file)

        self.test_dataset = PLYSingleCloudDataset(
            root=self._data_path,
            ply_path=ply_path,
            transform=self.test_transform,
            pre_transform=self.pre_transform,
        )

        self.train_dataset = EmptyDataset(self._data_path)
        self.val_dataset = EmptyDataset(self._data_path)

    def get_tracker(self, wandb_log=False, tensorboard_log=False):
        return SegmentationTracker(self, wandb_log=False, use_tensorboard=False)