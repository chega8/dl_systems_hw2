import numpy as np
from .autograd import Tensor
import gzip

from typing import Iterator, Optional, List, Sized, Union, Iterable, Any


class Transform:
    def __call__(self, x):
        raise NotImplementedError


class RandomFlipHorizontal(Transform):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        """
        Horizonally flip an image, specified as n H x W x C NDArray.
        Args:
            img: H x W x C NDArray of an image
        Returns:
            H x W x C ndarray corresponding to image flipped with probability self.p
        Note: use the provided code to provide randomness, for easier testing
        """
        flip_img = np.random.rand() < self.p
        ### BEGIN YOUR SOLUTION
        if flip_img:
            return np.fliplr(img)

        return img
        ### END YOUR SOLUTION


class RandomCrop(Transform):
    def __init__(self, padding=3):
        self.padding = padding

    def __call__(self, img):
        """Zero pad and then randomly crop an image.
        Args:
             img: H x W x C NDArray of an image
        Return
            H x W x C NAArray of cliped image
        Note: generate the image shifted by shift_x, shift_y specified below
        """
        shift_x, shift_y = np.random.randint(
            low=-self.padding, high=self.padding + 1, size=2
        )
        ### BEGIN YOUR SOLUTION
        h, w, c = img.shape
        padded_img = img
        pad_h = np.zeros((np.abs(shift_x), w, c))
        if shift_x < 0:
            padded_img = np.vstack((pad_h, padded_img))
            padded_img = padded_img[:shift_x, :]
        else:
            padded_img = np.vstack((img, pad_h))
            padded_img = padded_img[shift_x:, :]

        pad_w = np.zeros((h, np.abs(shift_y), c))
        if shift_y < 0:
            padded_img = np.hstack((pad_w, padded_img))
            padded_img = padded_img[:, :shift_y]
        else:
            padded_img = np.hstack((padded_img, pad_w))
            padded_img = padded_img[:, shift_y:]
        return padded_img
        ### END YOUR SOLUTION


class Dataset:
    r"""An abstract class representing a `Dataset`.

    All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
    data sample for a given key. Subclasses must also overwrite
    :meth:`__len__`, which is expected to return the size of the dataset.
    """

    def __init__(self, transforms: Optional[List] = None):
        self.transforms = transforms

    def __getitem__(self, index) -> object:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def apply_transforms(self, x):
        if self.transforms is not None:
            # apply the transforms
            for tform in self.transforms:
                x = tform(x)
        return x


class DataLoader:
    r"""
    Data loader. Combines a dataset and a sampler, and provides an iterable over
    the given dataset.
    Args:
        dataset (Dataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load
            (default: ``1``).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: ``False``).
    """
    dataset: Dataset
    batch_size: Optional[int]

    def __init__(
        self,
        dataset: Dataset,
        batch_size: Optional[int] = 1,
        shuffle: bool = False,
    ):

        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size
        if not self.shuffle:
            self.ordering = np.array_split(
                np.arange(len(dataset)), range(batch_size, len(dataset), batch_size)
            )

    def __iter__(self):
        ### BEGIN YOUR SOLUTION
        self.counter = 0
        if self.shuffle:
            idxs = np.arange(len(self.dataset))
            np.random.shuffle(idxs)
            self.ordering = np.array_split(
                idxs,
                range(self.batch_size, len(self.dataset), self.batch_size),
            )
        ### END YOUR SOLUTION
        return self

    def __next__(self):
        ### BEGIN YOUR SOLUTION
        if self.counter < len(self.ordering):
            batch_idx = self.ordering[self.counter]
            batch = self.dataset[batch_idx]
            self.counter += 1
            return tuple([Tensor(b) for b in batch])
        else:
            raise StopIteration
        ### END YOUR SOLUTION


class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        ### BEGIN YOUR SOLUTION
        self.image_filename = image_filename
        self.label_filename = label_filename
        super().__init__(transforms)

        num_images = 60000
        with gzip.open(image_filename) as bytestream:
            bytestream.read(16)
            buf = bytestream.read(28 * 28 * num_images)
            data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
            self.data = data.reshape(-1, 28, 28, 1)
            self.data = self.data / 255.0

        with gzip.open(label_filename) as bytestream:
            bytestream.read(8)
            buf = bytestream.read(1 * num_images)
            self.labels = np.frombuffer(buf, dtype=np.uint8)
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        ### BEGIN YOUR SOLUTION
        x = self.data[index]
        if self.transforms is not None:
            x = self.apply_transforms(x)
        return x, self.labels[index]
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        return len(self.labels)
        ### END YOUR SOLUTION


class NDArrayDataset(Dataset):
    def __init__(self, *arrays):
        self.arrays = arrays

    def __len__(self) -> int:
        return self.arrays[0].shape[0]

    def __getitem__(self, i) -> object:
        return tuple([a[i] for a in self.arrays])
