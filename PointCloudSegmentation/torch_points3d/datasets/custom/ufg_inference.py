import os
import torch
import numpy as np
from plyfile import PlyData
from torch_points3d.datasets.base_dataset import BaseDataset
from torch_points3d.metrics.segmentation_tracker import SegmentationTracker
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.nn import knn_interpolate
from torch_points3d.metrics.panoptic_tracker import PanopticResults


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
        plydata = PlyData.read(self.ply_path)
        vertices = plydata["vertex"]

        pts = np.vstack([vertices["x"], vertices["y"], vertices["z"]]).T
        pts = torch.tensor(pts, dtype=torch.float) # [N, 3]

        N = pts.shape[0]

        fake_feat = torch.ones((N, 1), dtype=torch.float)  # [N, 1]
        feats = torch.cat([pts, fake_feat], dim=1)  # [N, 4]

        data = Data(
            pos=pts,
            x=feats,
            y=torch.zeros(N, dtype=torch.long),
            instance_labels=torch.zeros(N, dtype=torch.long),
            center_label=torch.zeros(N, dtype=torch.long),
            num_classes=self.num_classes_value,
            num_instances=100,
            instance_mask=torch.zeros(N, dtype=torch.bool),
            vote_label=torch.zeros(N, 3, dtype=torch.float),
            original_id=torch.arange(N) 
        )

        # Wrap it in a list, as PyG expects a list of Data objects
        data_list = [data]

        if self.transform is not None:
            data_list = [data for data in data_list if self.transform(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        # Save in-memory dataset
        self.save(data_list, self.processed_paths[0])


class PlyInferenceDataset(BaseDataset):
    def __init__(self, dataset_opt):
        super().__init__(dataset_opt)

        ply_path = os.path.join(self._data_path, dataset_opt.ply_file)

        self.train_dataset = PlyInference(
            root=self._data_path,
            ply_path=ply_path,
            transform=None,
        )

        self.test_dataset = PlyInference(
            root=self._data_path,
            ply_path=ply_path,
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
    
    def predict_original_samples(self, batch, conv_type, output):
        """
        Upsample predictions to the original points of a single PLY file.

        Args:
            batch: processed Data object from the model
            conv_type: type of convolution ('DENSE', etc.)
            output: model predictions (torch.Tensor or PanopticResults from 3-head model)

        Returns:
            dict: {filename -> np.array([N, 4])} containing XYZ + predicted label
        """
        full_res_results = {}

        # Extract semantic logits
        if hasattr(output, "semantic_logits"):
            output_tensor = output.semantic_logits  # [N, num_classes]
            print("Output shape:", output_tensor.shape)
        elif isinstance(output, torch.Tensor):
            output_tensor = output
        else:
            raise TypeError(f"Expected torch.Tensor or PanopticResults with 'semantic_logits', got {type(output)}")

        # Original points
        sample_raw_pos = self.test_dataset[0].pos  # [N,3]

        # Move points to same device as output tensor
        device = output_tensor.device
        sample_raw_pos = sample_raw_pos.to(device)

        # Dense conv reshape
        if conv_type == "DENSE":
            output_tensor = output_tensor.reshape(1, -1, output_tensor.shape[-1])
            predicted = output_tensor[0]
        else:
            predicted = output_tensor

        origindid = batch.original_id  # tensor of indices of original points

        # Upsample predictions to original resolution
        print("Upsampling predictions...")
        full_prediction = knn_interpolate(predicted, sample_raw_pos[origindid], sample_raw_pos, k=3)

        # Convert logits/features to class labels
        print("Converting logits to class labels...")
        labels = full_prediction.max(1)[1].unsqueeze(-1)

        # Use filename as key
        filename = self.test_dataset[0].raw_file_names[0]

        # Save xyz + predicted label
        full_res_results[filename] = np.hstack((sample_raw_pos.cpu().numpy(), labels.cpu().numpy()))

        return full_res_results

