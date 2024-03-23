import os
from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from transformers import T5Tokenizer

from config import Config
from data import DInterface
from data.dataset import StageOneDataset, CaptionDataset, SearchDataset
from model import MInterface
from model.caption_model import T5ForImageCaptioning
from model.narrative_generation_model import T5ForNarrativeGeneration
from model.search_model import T5ForImageSearch
from utils.callbacks import UniversalCheckpoint

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def find_all_file(base: str, is_full_name=True):
    for root, ds, fs in os.walk(base):
        for f in fs:
            if not f.endswith(".ckpt"):
                continue
            if is_full_name:
                fullname = os.path.join(root, f)
                yield fullname
            else:
                yield f


def get_model(stage, task):
    if stage == 1:
        return T5ForNarrativeGeneration
    elif task == "caption":
        return T5ForImageCaptioning
    elif task == "search":
        return T5ForImageSearch


def get_dataset(stage, task):
    if stage == 1:
        return StageOneDataset
    elif task == "caption":
        return CaptionDataset
    elif task == "search":
        return SearchDataset


def main(args):
    pl.seed_everything(args.seed)

    tokenizer = T5Tokenizer.from_pretrained(args.text_model_path)
    if args.stage == 1:
        additional_special_tokens = ["[GENDER]"]
        tokenizer.add_special_tokens({"additional_special_tokens": additional_special_tokens})

    dataset = get_dataset(args.stage, args.task)
    data_module = DInterface(args, tokenizer, stage=args.stage, task_dataset=dataset)

    logger = TensorBoardLogger(save_dir=args.log_dir, name=args.dataset + "_" + args.task)

    callbacks = [UniversalCheckpoint(args)]

    trainer = Trainer.from_argparse_args(args, callbacks=callbacks, logger=logger)

    model = get_model(args.stage, args.task)
    model_interface = MInterface(args, tokenizer, model)
    model_interface.num_data = len(data_module.train_dataset)

    if args.do_train:
        trainer.fit(model_interface, data_module)

    if args.do_test:
        ckpt = ""  # ckpt file path
        args.ckpt_path = ckpt
        trainer.test(model_interface, data_module.test_dataloader(), ckpt_path=args.ckpt_path)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_test', action='store_true')

    args = parser.parse_args()
    args = Config(args)

    print(args)

    main(args)
