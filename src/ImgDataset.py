import numpy as np
import torch.utils.data
from PIL import Image
import torch
from torchvision import transforms
from pathlib import Path

class MultiviewImgDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, test_mode=False, num_models=0, num_views=12, shuffle=True):
        self.class_names = [
            'airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car', 'chair',
            'cone', 'cup', 'curtain', 'desk', 'door', 'dresser', 'flower_pot', 'glass_box',
            'guitar', 'keyboard', 'lamp', 'laptop', 'mantel', 'monitor', 'night_stand', 'person',
            'piano', 'plant', 'radio', 'range_hood', 'sink', 'sofa', 'stairs', 'stool', 'table',
            'tent', 'toilet', 'tv_stand', 'vase', 'wardrobe', 'xbox'
        ]
        self.root_dir = Path(root_dir)
        self.test_mode = test_mode
        self.num_views = num_views
        self.num_models = num_models

        self.filepaths = []

        for item in self.class_names:
            class_dir = self.root_dir / item
            if not class_dir.exists():
                continue

            model_dirs = sorted([p for p in class_dir.iterdir() if p.is_dir()])
            if self.num_models > 0:
                model_dirs = model_dirs[:min(self.num_models, len(model_dirs))]

            for model_dir in model_dirs:
                view_paths = sorted([str(p) for p in model_dir.glob('*.png')])
                if len(view_paths) < self.num_views:
                    print(f"Skipping {model_dir}, found {len(view_paths)} views (need {self.num_views})")
                    continue

                idxs = np.linspace(0, len(view_paths) - 1, self.num_views).astype(int)
                sampled = [view_paths[i] for i in idxs]
                self.filepaths.append(sampled)

        if shuffle:
            np.random.shuffle(self.filepaths)

        if self.test_mode:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        paths = self.filepaths[idx]
        class_name = Path(paths[0]).parents[1].name  # root/class/object/file.png
        class_id = self.class_names.index(class_name)
        imgs = []
        for path in paths:
            im = Image.open(path).convert('RGB')
            im = self.transform(im)
            imgs.append(im)
        return (class_id, torch.stack(imgs), paths)


class SingleImgDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, test_mode=False, num_models=0, num_views=12):
        self.class_names = [
            'airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car', 'chair',
            'cone', 'cup', 'curtain', 'desk', 'door', 'dresser', 'flower_pot', 'glass_box',
            'guitar', 'keyboard', 'lamp', 'laptop', 'mantel', 'monitor', 'night_stand', 'person',
            'piano', 'plant', 'radio', 'range_hood', 'sink', 'sofa', 'stairs', 'stool', 'table',
            'tent', 'toilet', 'tv_stand', 'vase', 'wardrobe', 'xbox'
        ]
        self.root_dir = Path(root_dir)
        self.test_mode = test_mode

        self.filepaths = []
        for item in self.class_names:
            class_dir = self.root_dir / item
            if not class_dir.exists():
                continue
            all_files = sorted([str(p) for p in class_dir.rglob('*.png')])
            if num_models == 0:
                self.filepaths.extend(all_files)
            else:
                self.filepaths.extend(all_files[:min(num_models * num_views, len(all_files))])

        if self.test_mode:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        path = self.filepaths[idx]
        class_name = Path(path).parents[1].name  # root/class/object/file.png
        class_id = self.class_names.index(class_name)
        im = Image.open(path).convert('RGB')
        im = self.transform(im)
        return (class_id, im, path)