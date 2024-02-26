from typing import Any, Dict, Tuple
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy
from torchmetrics.text.bleu import BLEUScore
from torchmetrics.text.bert import BERTScore
from torchmetrics.text.rouge import ROUGEScore
from torchmetrics.text.perplexity import Perplexity
from transformers import get_cosine_schedule_with_warmup
from transformers import InstructBlipProcessor, BertConfig, AutoProcessor, AlbertConfig
from sentence_transformers import SentenceTransformer, util
from peft import LoraConfig, get_peft_model

from .components.xinstructblip import InstructBlipForConditionalGeneration
from .components.xblip2 import Blip2ForConditionalGeneration
from .components.xropebert import RopeBertModel
from .components.xropealbert import RopeAlbertModel
from .components.xsampler import TemporalOFEmbedding
from .components.xraft import RAFT, InputPadder
from .components.raft_utils.utils import dp_state_to_normal
from src.gadgets.my_metrics import rouge_n, IoU

class LSTPSFModule(LightningModule):
    """a `LightningModule` for InstructBlip.

    A `LightningModule` implements 8 key methods:

    ```python
    def __init__(self):
    # Define initialization code here.

    def setup(self, stage):
    # Things to setup before each stage, 'fit', 'validate', 'test', 'predict'.
    # This hook is called on every process when using DDP.

    def training_step(self, batch, batch_idx):
    # The complete training step.

    def validation_step(self, batch, batch_idx):
    # The complete validation step.

    def test_step(self, batch, batch_idx):
    # The complete test step.

    def predict_step(self, batch, batch_idx):
    # The complete predict step.

    def configure_optimizers(self):
    # Define and configure optimizers and LR schedulers.
    ```

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        model_name_or_path: str,
        sampler_name_or_path: str,
        of_extractor_name_or_path: str,
        temperature: float,
        optimizer: torch.optim.Optimizer,
        # scheduler: torch.optim.lr_scheduler,
        scheduler: str,
        scheduler_params: dict,
        generate_configs: dict,
    ) -> None:
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        
        # hyperparameter
        self.temperature = temperature
        self.generate_configs = generate_configs

        # load model and init weight
        self.model = Blip2ForConditionalGeneration.from_pretrained(model_name_or_path)
        self.processor = AutoProcessor.from_pretrained(model_name_or_path)
        # if self.model.config.pad_token_id == -1:
        #     self.model.config.pad_token_id = self.processor.tokenizer.pad_token_id
        config = BertConfig(fusion_layer=6, encoder_width=768)
        self.temporal_encoder = RopeBertModel.from_pretrained(sampler_name_or_path, config=config)
        # config = AlbertConfig(fusion_layer=6, encoder_width=768)
        # self.temporal_encoder = RopeAlbertModel.from_pretrained(sampler_name_or_path, config=config)
        self.of_extractor = RAFT()
        state_dict = torch.load(of_extractor_name_or_path, map_location='cpu')
        state_dict = dp_state_to_normal(state_dict)
        msg = self.of_extractor.load_state_dict(state_dict)
        print(">>> Load checkpoint for of extractor from", sampler_name_or_path)
        miss = set(m.split('.')[0] for m in msg.missing_keys)
        unexp = set(m.split('.')[0] for m in msg.unexpected_keys)
        print("Missing:", miss if len(miss) else "None")
        print("Unexpected:", unexp if len(unexp) else "None")
        
        # loss function
        self.criterion = nn.CrossEntropyLoss()

        # metric objects for calculating and averaging score across batches
        self.train_iou_score = IoU()
        self.val_iou_score = IoU()
        self.test_bleu_score = IoU()

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation score
        self.val_score_best = MaxMetric()
        
        self.freeze_weights()

    def forward(self, batch):
        
        batch_size = batch["answer"].size(0)
        nframe = batch["nframe"]
        pixel_values = batch['frames']
        num_frames = pixel_values.size(0) // batch_size
        pixel_values = pixel_values.view(batch_size, num_frames, pixel_values.size(1), pixel_values.size(2), pixel_values.size(3))

        # OF encoder
        # of = []
        # for frames in pixel_values:
        #     padder = InputPadder(frames[0].shape)
        #     frames = padder.pad(frames)
        #     frames_feats = self.of_extractor(frames[:-1], frames[1:])
        #     frames_feats = torch.cat([frames_feats, frames_feats[-1].unsqueeze(0)], dim=0) # repeat last of
        #     of.append(frames_feats)
        # of = torch.stack(of)
        # of_mask = torch.ones(of.size(0), of.size(1)+2, dtype=torch.long, device=of.device)
        
        # input of
        of = batch["of"]
        of_mask = batch["of_mask"]
        # # input of_rgb
        # of = batch["of_rgb"]
        # of_mask = batch["of_rgb_mask"]
        
        # # self-refinement
        # # 1) get pseudo labels
        # with torch.no_grad():
        #     # TODO: use loop to prevent from OOM
        #     image_embeddings = self.model.vision_model(
        #         pixel_values=batch["frames"],
        #         return_dict=True
        #     ).last_hidden_state
        #     # image_attention_mask = torch.ones(image_embeddings.size()[:-1], dtype=torch.long, device=batch["frames"].device)
            
        #     image_attention_masks = torch.ones(image_embeddings.size()[:-1], dtype=torch.long, device=image_embeddings.device)
        #     query_tokens = self.model.query_tokens.expand(image_embeddings.shape[0], -1, -1)
        #     query_output = self.model.qformer(
        #         query_embeds=query_tokens,
        #         encoder_hidden_states=image_embeddings,
        #         encoder_attention_mask=image_attention_masks
        #     )[0]
                
        #     language_model_inputs = self.model.language_projection(query_output)
        #     language_attention_mask = torch.ones(
        #         language_model_inputs.size()[:-1], dtype=torch.long, device=language_model_inputs.device
        #     )
        #     question_attention_mask = torch.repeat_interleave(batch["question_attention_mask"], query_output.shape[0]//batch_size, 0)
        #     question = torch.repeat_interleave(batch["question"], query_output.shape[0]//batch_size, 0)
        #     attention_mask = torch.cat([language_attention_mask, question_attention_mask], dim=1)
        #     inputs_embeddings = self.model.get_input_embeddings()(question)
        #     # attention_mask = torch.cat([language_attention_mask, batch["question_attention_mask"]], dim=1)
        #     # inputs_embeddings = self.model.get_input_embeddings()(batch["question"])
        #     inputs_embeddings = torch.cat([language_model_inputs, inputs_embeddings], dim=1)
            
        #     outputs = self.model.language_model.generate(
        #         inputs_embeds=inputs_embeddings,
        #         attention_mask=attention_mask,
        #         max_length=128,
        #     )
        #     if self.model.config.text_config.architectures[0] == "LLaMAForCausalLM":
        #         outputs[outputs==0] = 2
        #     predict = self.processor.batch_decode(outputs, skip_special_tokens=True)
            
        #     ## evaluate results -> pseudo labels: better bleu score means better understand of the results
        #     target = batch["text_answer"]
        #     target = [target[int(idx//num_frames)] for idx in range(len(predict))]
        #     scores = rouge_n(target, predict)
        #     scores = torch.tensor(scores, dtype=torch.float)
        #     scores = scores.view(batch_size, num_frames)
        #     # monotone stack -> pseudo span
        #     start_targets = []
        #     end_targets = []
        #     for score in scores:
        #         bs = 0
        #         start_target = 0
        #         end_target = len(score) - 1
        #         stack = []
        #         score = [0] + score.tolist() + [0]
        #         for i in range(len(score)):
        #             while stack and score[stack[-1]] > score[i]:
        #                 tmp = stack.pop()
        #                 tmp_bs = (i-stack[-1]-1) * score[tmp]
        #                 if tmp_bs > bs:
        #                     bs = tmp_bs
        #                     start_target, end_target = stack[-1], i-2
        #             stack.append(i)
        #         start_targets.append(start_target)
        #         end_targets.append(end_target)
        #     # start_targets = torch.tensor(start_targets, dtype=torch.long, device=pixel_values.device)
        #     # end_targets = torch.tensor(end_targets, dtype=torch.long, device=pixel_values.device)
            
        #     flow_lengths = batch["of_lengths"]
        #     start_targets = [math.ceil((start_targets[ii]+1)/num_frames*flow_lengths[ii])-1  for ii in range(batch_size)]
        #     end_targets = [math.ceil((end_targets[ii]+1)/num_frames*flow_lengths[ii])-1  for ii in range(batch_size)]
            
        #     # try:
        #     #     for start_target in start_targets:
        #     #         assert start_target < of.size(1)
        #     #         assert start_target >= 0
        #     #     for end_target in end_targets:
        #     #         assert end_target < of.size(1)
        #     #         assert end_target >= 0
        #     # except:
        #     #     print(start_targets)
        #     #     print(of.size())
        #     #     raise ValueError("debug")

        #     start_targets = torch.tensor(start_targets, dtype=torch.long, device=pixel_values.device)
        #     end_targets = torch.tensor(end_targets, dtype=torch.long, device=pixel_values.device)

        #     # select_frames_idx = torch.topk(scores, nframe, dim=-1).indices.tolist()
        #     # refine_pos = []
        #     # for pos in select_frames_idx:
        #     #     refine_pos.append(sorted(pos))

        # 2) optimize temporal encoder
        of_feat, of_logits = self.temporal_encoder(
            encoder_embeds=of,
            attention_mask=of_mask,
            encoder_hidden_states=batch["sampler_question"],
            encoder_attention_mask=batch["sampler_question_attention_mask"],
            mode="multi_modal" # text/vision, fusion, multi_modal
        )

        start_logits, end_logits = of_logits.split(1, dim=-1)
        ignored_index = start_logits.size(1)
        
        # position label
        start_targets = batch["starts"]
        end_targets = batch["ends"]

        # mutli-gpu
        if len(start_targets.size()) > 1:
            start_targets = start_targets.squeeze(-1)
        if len(end_targets.size()) > 1:
            end_targets = end_targets.squeeze(-1)
        start_targets = start_targets.clamp(0, ignored_index)
        end_targets = end_targets.clamp(0, ignored_index)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()
        loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
        start_loss = loss_fct(start_logits, start_targets)
        end_loss = loss_fct(end_logits, end_targets)
        mrc_loss = (start_loss + end_loss) / 2
        
        return mrc_loss, start_logits, end_logits, start_targets, end_targets
    
    @torch.no_grad()
    def eval_forward(self, batch):
        batch_size = batch["answer"].shape[0]
        nframe = batch["nframe"]
        pixel_values = batch['frames']
        num_frames = pixel_values.size(0) // batch_size
        pixel_values = pixel_values.view(batch_size, num_frames, pixel_values.size(1), pixel_values.size(2), pixel_values.size(3))

        # # get pseudo label
        # # 1) get pseudo labels
        # with torch.no_grad():
        #     # TODO: use loop to prevent from OOM
        #     image_embeddings = self.model.vision_model(
        #         pixel_values=batch["frames"],
        #         return_dict=True
        #     ).last_hidden_state
        #     # image_attention_mask = torch.ones(image_embeddings.size()[:-1], dtype=torch.long, device=batch["frames"].device)
            
        #     image_attention_masks = torch.ones(image_embeddings.size()[:-1], dtype=torch.long, device=image_embeddings.device)
        #     query_tokens = self.model.query_tokens.expand(image_embeddings.shape[0], -1, -1)
        #     query_output = self.model.qformer(
        #         query_embeds=query_tokens,
        #         encoder_hidden_states=image_embeddings,
        #         encoder_attention_mask=image_attention_masks
        #     )[0]
                
        #     language_model_inputs = self.model.language_projection(query_output)
        #     language_attention_mask = torch.ones(
        #         language_model_inputs.size()[:-1], dtype=torch.long, device=language_model_inputs.device
        #     )
        #     question_attention_mask = torch.repeat_interleave(batch["question_attention_mask"], query_output.shape[0]//batch_size, 0)
        #     question = torch.repeat_interleave(batch["question"], query_output.shape[0]//batch_size, 0)
        #     attention_mask = torch.cat([language_attention_mask, question_attention_mask], dim=1)
        #     inputs_embeddings = self.model.get_input_embeddings()(question)
        #     # attention_mask = torch.cat([language_attention_mask, batch["question_attention_mask"]], dim=1)
        #     # inputs_embeddings = self.model.get_input_embeddings()(batch["question"])
        #     inputs_embeddings = torch.cat([language_model_inputs, inputs_embeddings], dim=1)
            
        #     outputs = self.model.language_model.generate(
        #         inputs_embeds=inputs_embeddings,
        #         attention_mask=attention_mask,
        #         max_length=128,
        #     )
        #     if self.model.config.text_config.architectures[0] == "LLaMAForCausalLM":
        #         outputs[outputs==0] = 2
        #     predict = self.processor.batch_decode(outputs, skip_special_tokens=True)
            
        #     ## evaluate results -> pseudo labels: better bleu score means better understand of the results
        #     target = batch["text_answer"]
        #     target = [target[int(idx//num_frames)] for idx in range(len(predict))]
        #     scores = rouge_n(target, predict)

        #     # print("============================================")
        #     # print(predict)
        #     # print(target)
        #     # print(scores)
        #     # print("============================================")

        #     scores = torch.tensor(scores, dtype=torch.float)
        #     scores = scores.view(batch_size, num_frames)
        #     # monotone stack -> pseudo span
        #     start_targets = []
        #     end_targets = []
        #     for score in scores:
        #         bs = 0
        #         start_target = 0
        #         end_target = len(score) - 1
        #         stack = []
        #         score = [0] + score.tolist() + [0]
        #         for i in range(len(score)):
        #             while stack and score[stack[-1]] > score[i]:
        #                 tmp = stack.pop()
        #                 tmp_bs = (i-stack[-1]-1) * score[tmp]
        #                 if tmp_bs > bs:
        #                     bs = tmp_bs
        #                     start_target, end_target = stack[-1], i-2
        #             stack.append(i)
        #         start_targets.append(start_target)
        #         end_targets.append(end_target)
        #     # start_targets = torch.tensor(start_targets, dtype=torch.long, device=pixel_values.device)
        #     # end_targets = torch.tensor(end_targets, dtype=torch.long, device=pixel_values.device)

        #     # process targets for dynamic length
        #     flow_lengths = batch["of_lengths"]
        #     # print(start_targets, len(start_targets))
        #     # print(predict, len(predict))
        #     # new_start_targets = []
        #     # for ii in range(len(predict)):
        #     #     start_target = start_targets[ii]
        #     #     ratio = start_target / num_frames
        #     #     flow_length = flow_lengths[ii // num_frames]
        #     #     new_start_target = int(ratio * flow_length + 0.5) - 1
        #     #     new_start_targets.append(new_start_target)
        #     # start_targets = new_start_targets

        #     start_targets = [math.ceil((start_targets[ii]+1)/num_frames*flow_lengths[ii])-1  for ii in range(batch_size)]
        #     end_targets = [math.ceil((end_targets[ii]+1)/num_frames*flow_lengths[ii])-1  for ii in range(batch_size)]
            
        #     start_targets = torch.tensor(start_targets, dtype=torch.long, device=pixel_values.device)
        #     end_targets = torch.tensor(end_targets, dtype=torch.long, device=pixel_values.device)
        
        start_targets = batch["starts"]
        end_targets = batch["ends"]
        
        # 0. get OF embeeding from temporal encoder
        # of = []
        # for frames in pixel_values:
        #     padder = InputPadder(frames[0].shape)
        #     frames = padder.pad(frames)
        #     frames_feats = self.of_extractor(frames[:-1], frames[1:])
        #     frames_feats = torch.cat([frames_feats, frames_feats[-1].unsqueeze(0)], dim=0) # repeat last of
        #     of.append(frames_feats)
        # of = torch.stack(of)
        # of_mask = torch.ones(of.size(0), of.size(1)+2, dtype=torch.long, device=of.device)
        
        # input flow
        of = batch["of"]
        of_mask = batch["of_mask"]
        # # input flow-rgb
        # of = batch["of_rgb"]
        # of_mask = batch["of_rgb_mask"]

        of_feat, of_logits = self.temporal_encoder(
            encoder_embeds=of,
            attention_mask=of_mask,
            encoder_hidden_states=batch["sampler_question"],
            encoder_attention_mask=batch["sampler_question_attention_mask"],
            mode="multi_modal" # text/vision: first 6 layers | fusion: last 6 layers | multi_modal: all 12 layers 
        )
        start_logits, end_logits = of_logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()
            
        return start_logits, end_logits, start_targets, end_targets

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        # self.val_rouge_score.reset()
        self.val_iou_score.reset()
        self.val_score_best.reset()

    def model_step(
        self, batch: Dict[str, torch.Tensor]
    ):
        
        loss, start_logits, end_logits, start_targets, end_targets = self.forward(batch)
        return loss, start_logits, end_logits, start_targets, end_targets
    
    def eval_model_step(
        self, batch
    ):
        start_logits, end_logits, start_targets, end_targets = self.eval_forward(batch)
        return start_logits, end_logits, start_targets, end_targets

    def training_step(
        self, batch, batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        loss, start_logits, end_logits, start_targets, end_targets = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True)

        iou_score, iou3_score, iou5_score = self.train_iou_score(start_logits, end_logits, start_targets, end_targets)
        self.log("train/iou_score", iou_score, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/iou3_score", iou3_score, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/iou5_score", iou5_score, on_step=True, on_epoch=True, prog_bar=True)
        
        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def validation_step(self, batch, batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        
        start_logits, end_logits, start_targets, end_targets = self.eval_model_step(batch)

        if batch_idx % 100 == 0:
            print("=============================================")
            print(start_logits.argmax(dim=-1))
            print(start_targets)
            print(end_logits.argmax(dim=-1))
            print(end_targets)
            print("=============================================")

        # update and log metrics
        iou_score, iou3_score, iou5_score = self.val_iou_score(start_logits, end_logits, start_targets, end_targets)
        self.log("val/iou_score", iou_score, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/iou3_score", iou3_score, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/iou5_score", iou5_score, on_step=False, on_epoch=True, prog_bar=True)


    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        # score = self.val_bert_score.compute()['f1']  # get current val score
        iou_score, iou3_score, iou5_score = self.val_iou_score.compute()
        self.val_score_best(iou_score)  # update best so far val score
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/score_best", self.val_score_best.compute(), sync_dist=True, prog_bar=True)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        
        start_logits, end_logits, start_targets, end_targets = self.eval_model_step(batch)

        # update and log metrics
        # self.test_rouge_score(preds, targets)
        iou_score, iou3_score, iou5_score = self.test_iou_score(start_logits, end_logits, start_targets, end_targets)
        self.log("test/iou_score", iou_score, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/iou3_score", iou3_score, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/iou5_score", iou5_score, on_step=False, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        pass

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configures optimizers and learning-rate schedulers to be used for training.

        Normally you'd need one, but in the case of GANs or similar you might need multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.parameters())
        # # # original scheduler
        # if self.hparams.scheduler is not None:
        #     scheduler = self.hparams.scheduler(optimizer=optimizer)
        #     return {
        #         "optimizer": optimizer,
        #         "lr_scheduler": {
        #             "scheduler": scheduler,
        #             "monitor": "val/score",
        #             "interval": "epoch",
        #             "frequency": 1,
        #         },
        #     }

        # rewrite from transformers.get_cosine_schedule_with_warmup
        if self.hparams.scheduler is not None:
            
            if self.hparams.scheduler == "cosine":
                max_steps = self.trainer.max_steps
                warmup_steps = int(max_steps * self.hparams.scheduler_params["warmup_steps"])
                scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=max_steps)
            else:
                raise NotImplementedError("UNKONWN SCHEDULER")

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/score",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }

        return {"optimizer": optimizer}

        
    def freeze_weights(self):
        for param in self.model.vision_model.parameters():
            param.requires_grad = False
        for param in self.model.language_model.parameters():
            param.requires_grad = False
        for param in self.model.qformer.parameters():
            param.requires_grad = False

    def dp_state_to_normal(state_dict):
        '''Converts a torch.DataParallel checkpoint to regular'''
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module'):
                new_state_dict[k.replace('module.', '')] = v
        return new_state_dict
            

if __name__ == "__main__":
    _ = LSTPSFModule(None, None, None)
