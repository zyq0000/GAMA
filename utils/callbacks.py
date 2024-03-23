from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import EarlyStopping


class UniversalEarlyStopping(EarlyStopping):
    def __init__(self, args):
        super().__init__(
            monitor=args.monitor,
            mode=args.mode,
            patience=args.patience
        )


class UniversalCheckpoint(ModelCheckpoint):
    def __init__(self, args):
        super().__init__(
            monitor=args.monitor,
            save_top_k=args.save_top_k,
            mode=args.mode,
            filename=f"{args.text_model}_{args.vision_model}_lr{args.learning_rate}_ep{args.max_epochs}_w{args.warm_factor}_tao{args.tao}" + args.ckpt_filename,
            save_last=False,
        )
