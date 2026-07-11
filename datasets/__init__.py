import os

from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import IMG_EXTENSIONS
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import transforms
import torch.distributed as dist


def _is_valid_image_file(path):
    """ I/O robustness filter only (does not change preprocessing, model,
    distance metric or loss). A handful of files shipped with the upstream
    GenImage release are pre-existing 0-byte/corrupted files (e.g. some
    BigGAN/ADM/Midjourney train images), unrelated to our 1/5 sampling.
    Without this filter, ImageFolder indexes them and the DataLoader worker
    crashes with PIL.UnidentifiedImageError as soon as one is sampled. """
    if not path.lower().endswith(IMG_EXTENSIONS):
        return False
    try:
        return os.path.getsize(path) > 0
    except OSError:
        return False


def setup_infinity_train_dataloader(
    folder_path, 
    batch_size=20, 
    num_workers=16, 
    pin_memory=True, 
    drop_last=True
):
    transform = transforms.Compose([
        transforms.Resize(256), 
        transforms.RandomCrop(224), 
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(),
    ])
    dataset = ImageFolder(folder_path, transform=transform, is_valid_file=_is_valid_image_file) # As GenImage dataset is very large, so it's ok for different size of each class
    sampler = DistributedSampler(dataset) if dist.is_initialized() else None

    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=(sampler is None), 
        sampler=sampler, 
        num_workers=num_workers, 
        pin_memory=pin_memory, 
        drop_last=drop_last
    )
    
    # adding epoch control
    epoch = 0

    while True: 
        if sampler is not None: 
            sampler.set_epoch(epoch)
        epoch += 1
        yield from loader


def setup_val_dataloader(
    folder_path, 
    batch_size=20, 
    num_workers=16, 
    pin_memory=True, 
    drop_last=True
): 
    transform = transforms.Compose([ 
        transforms.CenterCrop(224), 
        transforms.ToTensor(), 
    ])

    return DataLoader(
        ImageFolder(folder_path, transform=transform, is_valid_file=_is_valid_image_file), 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=pin_memory, 
        drop_last=drop_last, 
    )

