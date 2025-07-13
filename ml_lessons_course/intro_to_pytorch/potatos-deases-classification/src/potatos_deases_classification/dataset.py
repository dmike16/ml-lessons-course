from typing import Callable, List, Optional, Any
from torch.utils.data import Dataset
from torchvision.datasets.folder import find_classes
import os
import numpy as np
import skimage as ski


class PotatosDataset(Dataset):
    """Custom dataset implementation"""

    def __init__(self, root_dir: str, transform: Optional[Callable] = None):
        self.root_dir = root_dir
        self.transform = transform
        self._classes, self._class_to_idx = find_classes(root_dir)
        self._ds = make_dataset(root_dir, self._class_to_idx)

    @property
    def classes(self) -> List[str]:
        return self._classes

    @property
    def classes_to_idx(self) -> dict[str, int]:
        return self._class_to_idx

    def __len__(self) -> int:
        return len(self._ds)

    def __getitem__(self, index) -> tuple[Any, Any]:
        path, class_idx = self._ds[index]
        item = load(path)

        if self.transform is not None:
            item = self.transform(item)

        return item, class_idx


def load(path: str) -> np.ndarray:
    return ski.io.imread(path)


def make_dataset(root_dir: str,
                 class_to_idx: dict[str, int]) -> list[tuple[str, int]]:
    dir = os.path.expanduser(root_dir)
    instances = []
    for targe_class in sorted(class_to_idx.keys()):
        class_idx = class_to_idx[targe_class]
        target_dir = os.path.join(dir, targe_class)
        if not os.path.isdir(target_dir):
            continue
        for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                item = path, class_idx
                instances.append(item)

    return instances
