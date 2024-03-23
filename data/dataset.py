import os

import numpy as np
import torch
from torch.utils.data import Dataset

os.environ["TOKENIZERS_PARALLELISM"] = "false"


IMG_SHAPE_MODEL = {
    "vit-base": (1, 577, 768)
}


class StageOneDataset(Dataset):
    def __init__(self, args, dataset, tokenizer, stage):
        super().__init__()
        self.args = args
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.stage = stage

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        return self.encode(self.dataset[item])

    def encode(self, data):
        data_stage = data[str(self.stage)]
        source_text = "Task: " + data_stage["question"] + " Answer: "
        target_text = data_stage["answer"]

        target_text_2 = data_stage["answer_2"]

        source = self.tokenizer.encode_plus(source_text, max_length=self.args.max_input_length, padding="max_length",
                                            truncation=True, return_tensors="pt")
        target = self.tokenizer.encode_plus(target_text, max_length=self.args.max_output_length, padding="max_length",
                                            truncation=True, return_tensors="pt")

        target_2 = self.tokenizer.encode_plus(target_text_2, max_length=self.args.max_output_length,
                                              padding="max_length",
                                              truncation=True, return_tensors="pt")
        source_ids = source["input_ids"]
        source_mask = source["attention_mask"]
        target_ids = target["input_ids"]
        target_ids_2 = target_2["input_ids"]

        filename = data["image_path"].split("/")[-1].split(".")[0]
        feature_path = os.path.join(self.args.image_feature_dir, self.args.dataset, self.args.vision_model,
                                    filename + ".npy")
        if os.path.exists(feature_path):
            image_features = torch.Tensor(np.load(feature_path))
        else:
            print(feature_path)
            image_features = torch.zeros(IMG_SHAPE_MODEL[self.args.vision_model])

        return {
            "image_features": image_features,
            "input_ids": source_ids,
            "attention_mask": source_mask,
            "labels": target_ids,
            "labels_2": target_ids_2
        }


class CaptionDataset(Dataset):
    def __init__(self, args, dataset, tokenizer, stage):
        super().__init__()
        self.args = args
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.stage = stage

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        return self.encode(self.dataset[item])

    def encode(self, data):
        data_stage = data[str(self.stage)]
        # print(data_stage)
        source_text = "Context: " + data_stage["context"] + " Task: " + data_stage["question"] + " Answer: "
        # print(source_text)
        target_text = data_stage["answer"]

        source = self.tokenizer.encode_plus(source_text, max_length=self.args.max_input_length, padding="max_length",
                                            truncation=True, return_tensors="pt")
        target = self.tokenizer.encode_plus(target_text, max_length=self.args.max_output_length, padding="max_length",
                                            truncation=True, return_tensors="pt")

        source_ids = source["input_ids"]
        source_mask = source["attention_mask"]
        target_ids = target["input_ids"]

        filename = data["image_path"].split(".")[0]
        feature_path = os.path.join(self.args.image_feature_dir, self.args.dataset, self.args.vision_model,
                                    filename + ".npy")

        if os.path.exists(feature_path):
            image_features = torch.Tensor(np.load(feature_path))
        else:
            print(feature_path)
            image_features = torch.zeros(IMG_SHAPE_MODEL[self.args.vision_model])
        # image_features = self.all_image_features[filename]

        return {
            "image_features": image_features,
            "input_ids": source_ids,
            "attention_mask": source_mask,
            "labels": target_ids
        }


class SearchDataset(Dataset):
    def __init__(self, args, dataset, tokenizer, stage):
        super().__init__()
        self.args = args
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.stage = stage

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        return self.encode(self.dataset[item])

    def encode(self, data):
        data_stage = data[str(self.stage)]
        source_text = "Context: " + data_stage["context"] + " Query: " + data_stage["caption"] + " Question: " + \
                      data_stage["question"] + " Answer: "

        target_text = data_stage["answer"]

        source = self.tokenizer.encode_plus(source_text, max_length=self.args.max_input_length,
                                            padding="max_length", truncation=True, return_tensors="pt")
        target = self.tokenizer.encode_plus(target_text, max_length=self.args.max_output_length, padding="max_length",
                                            truncation=True, return_tensors="pt")

        source_ids = source["input_ids"]
        attention_mask = source["attention_mask"]

        target_ids = target["input_ids"]

        filename = data["image_path"].split(".")[0]
        feature_path = os.path.join(self.args.image_feature_dir, self.args.dataset, self.args.vision_model,
                                    filename + ".npy")

        if os.path.exists(feature_path):
            image_features = torch.Tensor(np.load(feature_path))
        else:
            print(feature_path)
            image_features = torch.zeros(IMG_SHAPE_MODEL[self.args.vision_model])

        return {
            "image_features": image_features,
            "input_ids": source_ids,
            "attention_mask": attention_mask,
            "labels": target_ids,
        }
