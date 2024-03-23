import os

import numpy as np
import pytorch_lightning as pl
import torch

from utils.data_utils import IMG_SHAPE, write_pred_data


class MInterface(pl.LightningModule):
    def __init__(self, args, tokenizer, task_model):
        super().__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.model = task_model.from_pretrained(
            self.args.text_model_path, patch_size=IMG_SHAPE[self.args.vision_model],
            padding_idx=self.tokenizer.pad_token_id, stage=self.args.stage, tao=self.args.tao)
        self.model.resize_token_embeddings(len(tokenizer))
        self.text_path = os.path.join(args.text_data_dir)

    def setup(self, stage=None) -> None:
        if stage == 'fit':
            num_gpus = self.trainer.gpus if self.trainer.gpus is not None else 0
            self.total_step = int(
                self.trainer.max_epochs * self.num_data / (max(1, num_gpus) * self.trainer.accumulate_grad_batches))
            print('Total training step:', self.total_step)

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        params = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        optimizer = torch.optim.AdamW(params, lr=self.args.learning_rate)

        return [{
            'optimizer': optimizer,
        }]

    def training_step(self, batch, batch_idx):
        labels = batch["labels"]
        labels[labels == self.tokenizer.pad_token_id] = -100

        if self.args.stage == 1:
            labels_2 = batch["labels_2"]
            labels_2[labels_2 == self.tokenizer.pad_token_id] = -100
            outputs = self.model(input_ids=batch["input_ids"], image_features=batch["image_features"],
                                 attention_mask=batch["attention_mask"], labels=labels, labels_2=labels_2,
                                 stage="train")
        elif self.args.task in ["caption", "search"]:
            outputs = self.model(input_ids=batch["input_ids"], image_features=batch["image_features"],
                                 attention_mask=batch["attention_mask"], labels=labels, stage="train")
        return outputs.loss

    def validation_step(self, batch, batch_idx):
        labels = batch["labels"]
        labels[labels == self.tokenizer.pad_token_id] = -100
        outputs = None
        if self.args.stage == 1:
            labels_2 = batch["labels_2"]
            labels_2[labels_2 == self.tokenizer.pad_token_id] = -100
            outputs = self.model(input_ids=batch["input_ids"], image_features=batch["image_features"],
                                 attention_mask=batch["attention_mask"], labels=labels, labels_2=labels_2, stage="val")
        elif self.args.task in ["caption", "search"]:
            outputs = self.model(input_ids=batch["input_ids"], image_features=batch["image_features"],
                                 attention_mask=batch["attention_mask"], labels=labels, stage="val")
        return {
            "loss": outputs.loss
        }

    def validation_epoch_end(self, val_step_outputs):
        avg_loss = torch.stack([x["loss"] for x in val_step_outputs]).mean()
        print(avg_loss)
        self.log("val_loss", avg_loss, prog_bar=True)

    def test_step(self, batch, batch_idx):
        if self.args.task == "search":
            outputs = self.model.generate(input_ids=batch["input_ids"], image_features=batch["image_features"],
                                          attention_mask=batch["attention_mask"],
                                          repetition_penalty=2.0, temperature=0.7, num_beams=5,
                                          max_length=self.args.max_output_length, stage="test",
                                          return_dict_in_generate=True, output_scores=True)
            sequences = self.tokenizer.batch_decode(
                outputs["sequences"].cpu().numpy(), skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            scores = np.exp(outputs["sequences_scores"].cpu().numpy())
            predictions = [{"ans": sequence, "score": float(score)} for sequence, score in zip(sequences, scores)]

        else:
            predictions = self.model.generate(input_ids=batch["input_ids"], image_features=batch["image_features"],
                                              attention_mask=batch["attention_mask"],
                                              repetition_penalty=2.0, temperature=0.7, num_beams=5,
                                              max_length=self.args.max_output_length, stage="test")
            predictions = predictions.cpu().numpy()
            predictions = self.tokenizer.batch_decode(
                predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
        return {
            "predictions": predictions
        }

    def test_epoch_end(self, test_step_outputs):
        predictions = [item for pred in test_step_outputs for item in pred['predictions']]

        print(predictions[0])
        write_pred_data(self.args, self.text_path, predictions)
