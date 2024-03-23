import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from utils.data_utils import load_data


class DInterface(pl.LightningDataModule):
    def __init__(self, args, tokenizer, stage, task_dataset):
        super().__init__()
        self.args = args

        self.tokenizer = tokenizer

        self.train_dataset, self.dev_dataset, self.test_dataset = load_data(args, args.text_data_dir)

        self.train_set = task_dataset(self.args, self.train_dataset, self.tokenizer, stage=stage)
        self.dev_set = task_dataset(self.args, self.dev_dataset, self.tokenizer, stage=stage)
        self.test_set = task_dataset(self.args, self.test_dataset, self.tokenizer, stage=stage)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.args.train_batch_size, num_workers=self.args.num_workers, collate_fn=self.collate_fn, shuffle=True, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.dev_set, batch_size=self.args.eval_batch_size, num_workers=self.args.num_workers, collate_fn=self.collate_fn, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.args.eval_batch_size, num_workers=self.args.num_workers, collate_fn=self.collate_fn, shuffle=False)

    def collate_fn(self, batch):
        batch_data = {}

        for key in batch[0]:
            batch_data[key] = torch.cat([item[key] for item in batch], dim=0)

        return batch_data
