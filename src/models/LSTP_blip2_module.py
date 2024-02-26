from typing import Any, Dict, Tuple
import math
from functools import partial

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
import transformers
from transformers import get_cosine_schedule_with_warmup
from transformers import AutoProcessor, BertConfig, InstructBlipProcessor
from sentence_transformers import SentenceTransformer, util

# from .components.xinstructblip import InstructBlipForConditionalGeneration
from .components.xblip2 import Blip2ForConditionalGeneration
from .components.xropebert import RopeBertModel
from .components.xsampler import TemporalOFEmbedding
from .components.xraft import RAFT, InputPadder
from .components.raft_utils.utils import dp_state_to_normal

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

class LSTPModule(LightningModule):
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
        scheduler: str,
        scheduler_params: dict,
        generate_configs: dict,
    ) -> None:
        super().__init__()
        # super(LSTPModule,self).__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        
        # hyperparameter
        self.temperature = temperature
        self.generate_configs = generate_configs

        # load model and init weight
        # self.model = InstructBlipForConditionalGeneration.from_pretrained(model_name_or_path)
        self.model = Blip2ForConditionalGeneration.from_pretrained(model_name_or_path)
        self.processor = AutoProcessor.from_pretrained(model_name_or_path)
        # if self.model.config.pad_token_id == -1: # vicuna1
        #     self.model.config.pad_token_id = self.processor.tokenizer.pad_token_id
        #     self.model.config.text_config.pad_token_id = self.processor.tokenizer.pad_token_id
        # self.model.config.text_config.pad_token_id = self.processor.tokenizer.pad_token_id
        # self.model.language_model.resize_token_embedding(len(self.processor.tokenizer))
        # if self.model.config.use_decoder_only_language_model:
        #     self.model.config.text_config.pad_token_id = self.processor.tokenizer.pad_token_id
        #     raise ValueError("debug")
        #     smart_tokenizer_and_embedding_resize(
        #         special_tokens_dict=dict(
        #             pad_token='[PAD]',
        #             bos_token='</s>',
        #             eos_token='</s>',
        #             unk_token='</s>'
        #         ),
        #         tokenizer=self.processor.tokenizer,
        #         model=self.model.language_model
        #     )
            # self.processor.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            # self.processor.tokenizer.add_special_tokens({'bos_token': '</s>'})
            # self.processor.tokenizer.add_special_tokens({'eos_token': '</s>'})
            # self.processor.tokenizer.add_special_tokens({'unk_token': '</s>'})
            # self.processor.qformer_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            # self.processor.qformer_tokenizer.add_special_tokens({'bos_token': '</s>'})
            # self.processor.qformer_tokenizer.add_special_tokens({'eos_token': '</s>'})
            # self.processor.qformer_tokenizer.add_special_tokens({'unk_token': '</s>'})
            # self.model.language_model.resize_token_embeddings(len(self.processor.tokenizer))
            # self.model.qformer.resize_token_embeddings(len(self.processor.tokenizer))
        config = BertConfig(fusion_layer=6, encoder_width=768)
        self.temporal_encoder = RopeBertModel.from_pretrained(sampler_name_or_path, config=config)
        # # load from pretrained sampler
        # config = BertConfig(fusion_layer=6, encoder_width=768)
        # self.temporal_encoder = RopeBertModel(config=config)
        # ckpt = torch.load(sampler_name_or_path, map_location="cpu")
        # state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
        # msg = self.temporal_encoder.load_state_dict(state_dict, strict=False)
        # print(">>> Load checkpoint for sampler from", sampler_name_or_path)
        # miss = set(m.split('.')[0] for m in msg.missing_keys)
        # unexp = set(m.split('.')[0] for m in msg.unexpected_keys)
        # print("Missing:", miss if len(miss) else "None")
        # print("Unexpected:", unexp if len(unexp) else "None")
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
        # self.train_score = Perplexity()
        # self.val_rouge_score = ROUGEScore()
        # self.test_rouge_score = ROUGEScore()
        self.val_bleu_score = BLEUScore(n_gram=1)
        self.test_bleu_score = BLEUScore(n_gram=1)
        # self.val_bert_score = BERTScore()
        # self.test_bert_score = BERTScore()

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
        
        # # 0. get OF embedding from temporal encoder
        # # 0.0 get OF feature from RAFT
        # of = []
        # for frames in pixel_values:
        #     padder = InputPadder(frames[0].shape)
        #     frames = padder.pad(frames)
        #     frames_feats = self.of_extractor(frames[:-1], frames[1:])
        #     frames_feats = torch.cat([frames_feats, frames_feats[-1].unsqueeze(0)], dim=0) # repeat last of
        #     of.append(frames_feats)
        # of = torch.stack(of)
        # of_mask = torch.ones(of.size(0), of.size(1)+2, dtype=torch.long, device=of.device)
        # of_feat, of_logits = self.temporal_encoder(
        #     encoder_embeds=of,
        #     attention_mask=of_mask,
        #     # encoder_embeds=batch["of"],
        #     # attention_mask=batch["of_mask"],
        #     encoder_hidden_states=batch["sampler_question"],
        #     encoder_attention_mask=batch["sampler_question_attention_mask"],
        #     mode="fusion" # text/vision: first 6 layers | fusion: last 6 layers | multi_modal: all 12 layers 
        # )
        # start_logits, end_logits = of_logits.split(1, dim=-1)
        # # 0.1 sample image by OF indices
        # pixel_shape = list(pixel_values.size())
        # pixel_shape[1] = nframe
        # sampled_pixel_values = torch.zeros(pixel_shape, device=pixel_values.device)
        # cand_start_index = []
        # cand_end_index = []
        # top_k = 2
        # for _ in range(top_k):
        #     logits = torch.cat([start_logits, end_logits], dim=0)
        #     prob = F.gumbel_softmax(logits, tau=0.5, dim=1)
        #     # lb = torch.zeros(prob.shape).to(prob.device)
        #     # prob = torch.where(prob<0.85, lb, prob)
        #     index = prob.argmax(dim=1)
        #     cand_start_index.append(index[:batch_size])
        #     cand_end_index.append(index[batch_size:])
        pixel_shape = list(pixel_values.size())
        pixel_shape[1] = nframe
        sampled_pixel_values = torch.zeros(pixel_shape, device=pixel_values.device)
        video_lengths = batch["of_lengths"]
        
        # get frame idx
        for j in range(len(video_lengths)):
            video_length = video_lengths[j]
            # video_length = num_frames + 2
            
            # # # sample for LSTP
            # cand_index = set()
            # # gumbel softmax
            # for ii in range(len(cand_start_index)):
            #     cand_start = cand_start_index[ii][j]
            #     cand_end = cand_end_index[ii][j]
            #     if cand_start >= video_length or cand_end >= video_length or (cand_start == 0 and cand_end == 0):
            #         cand_start = 0
            #         cand_end = video_length - 1
            #     # if cand_end < cand_start: cand_end = cand_start
            #     start, end = int(cand_start/video_length*num_frames), int(cand_end/video_length*num_frames)
            #     cand_index = set.union(cand_index, set(list(range(start, end))))
            # cand_index = sorted(list(cand_index))
            
            # uniform for baseline
            cand_index = list(range(num_frames))
            
            if cand_index == []:
                cand_index = list(range(num_frames))
            while len(cand_index) < nframe:
                cand_index = [xx for x in cand_index for xx in (x, x)]
            if len(cand_index) > nframe:
                intv = np.linspace(start=0, stop=len(cand_index), num=nframe+1).astype(int)
                new_cand_index = [cand_index[(intv[x]+intv[x+1]-1)//2] for x in range(len(intv)-1)]
                # new_cand_index = [cand_index[random.randrange(intv[x],intv[x+1])] for x in range(len(intv)-1)]
                cand_index = new_cand_index
            assert len(cand_index) == nframe
            cand_index = torch.tensor(cand_index, device=pixel_values.device)
            sampled_pixel_values[j] = torch.index_select(pixel_values[j], 0, cand_index) 
        
        # 1. get image embedding from vision encoder
        sampled_pixel_values = sampled_pixel_values.view(batch_size*nframe, sampled_pixel_values.size(2), sampled_pixel_values.size(3), sampled_pixel_values.size(4))
        vision_outputs = self.model.vision_model(
            pixel_values=sampled_pixel_values
        )
        image_embeddings = vision_outputs[0]
        
        # 2. get qformer embedding
        image_attention_masks = torch.ones(image_embeddings.size()[:-1], dtype=torch.long, device=image_embeddings.device)
        query_tokens = self.model.query_tokens.expand(image_embeddings.shape[0], -1, -1)
        query_output = self.model.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeddings,
            encoder_attention_mask=image_attention_masks
        )[0]

        # 3. generate conditioned on query_output
        ## reshape qformer representation
        language_model_inputs = self.model.language_projection(query_output)
        language_model_inputs = language_model_inputs.reshape(batch_size, -1, language_model_inputs.shape[-1])
        language_model_attention_mask = torch.ones(
            language_model_inputs.size()[:-1], dtype=torch.long, device=language_model_inputs.device
        )
        # ## concatenate qformer representation and OF feature
        # of_feat = self.model.temporal_projection(of_feat)
        # of_feat_attn = torch.ones(
        #     of_feat.size()[:-1], dtype=torch.long, device=of_feat.device
        # )
        ## LLM
        if self.model.config.use_decoder_only_language_model:
            inputs_embeddings = self.model.language_model.get_input_embeddings()(batch["instruction"])
            # inputs_embeddings = torch.cat([of_feat, language_model_inputs, inputs_embeddings.to(language_model_inputs.device)], dim=1)
            # attention_mask = torch.cat([of_feat_attn, language_model_attention_mask.to(batch["instruction_attention_mask"]), batch["instruction_attention_mask"]], dim=1)
            inputs_embeddings = torch.cat([language_model_inputs, inputs_embeddings.to(language_model_inputs.device)], dim=1)
            attention_mask = torch.cat([language_model_attention_mask.to(batch["instruction_attention_mask"]), batch["instruction_attention_mask"]], dim=1)
            outputs = self.model.language_model(
                inputs_embeds=inputs_embeddings,
                attention_mask=attention_mask
            )
            logits = outputs[0]
            loss = None
            labels = batch["instruction"]
            logits = logits[:, -labels.size(1):, :]
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous().to(logits.device)
            # print("=================================================")
            # print(shift_logits)
            # print(shift_labels)
            # print("=================================================")
            loss_fct = CrossEntropyLoss(reduction="mean")
            loss = loss_fct(shift_logits.view(-1, self.model.config.text_config.vocab_size), shift_labels.view(-1))
        else:
            inputs_embeddings = self.model.language_model.get_input_embeddings()(batch["question"])
            # inputs_embeddings = torch.cat([of_feat, language_model_inputs, inputs_embeddings.to(language_model_inputs.device)], dim=1)
            # attention_mask = torch.cat([of_feat_attn, language_model_attention_mask.to(batch["question_attention_mask"]), batch["question_attention_mask"]], dim=1)
            inputs_embeddings = torch.cat([language_model_inputs, inputs_embeddings.to(language_model_inputs.device)], dim=1)
            attention_mask = torch.cat([language_model_attention_mask.to(batch["question_attention_mask"]), batch["question_attention_mask"]], dim=1)
            labels = batch["answer"].masked_fill(self.processor.tokenizer.pad_token_id == batch["answer"], -100)
            outputs = self.model.language_model(
                inputs_embeds=inputs_embeddings,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs[0]
            logits = outputs[1]
        
        return loss, logits
    
    @torch.no_grad()
    def eval_forward(self, batch):
        batch_size = batch["answer"].shape[0]
        nframe = batch["nframe"]
        pixel_values = batch['frames']
        num_frames = pixel_values.size(0) // batch_size
        pixel_values = pixel_values.view(batch_size, num_frames, pixel_values.size(1), pixel_values.size(2), pixel_values.size(3))
        
        # # 0. get OF embeeding from temporal encoder
        # # 0.0 get OF feature from RAFT
        # of = []
        # for frames in pixel_values:
        #     padder = InputPadder(frames[0].shape)
        #     frames = padder.pad(frames)
        #     frames_feats = self.of_extractor(frames[:-1], frames[1:])
        #     frames_feats = torch.cat([frames_feats, frames_feats[-1].unsqueeze(0)], dim=0) # repeat last of
        #     of.append(frames_feats)
        # of = torch.stack(of)
        # of_mask = torch.ones(of.size(0), of.size(1)+2, dtype=torch.long, device=of.device)
        # # print("===========================================")
        # # print(of.size())
        # # print(batch["of"].size())
        # # print(of_mask.size())
        # # print(batch["of_mask"].size())
        # # print("===========================================")
        # of_feat, of_logits = self.temporal_encoder(
        #     encoder_embeds=of,
        #     attention_mask=of_mask,
        #     # encoder_embeds=batch["of"],
        #     # attention_mask=batch["of_mask"],
        #     encoder_hidden_states=batch["sampler_question"],
        #     encoder_attention_mask=batch["sampler_question_attention_mask"],
        #     mode="fusion" # text/vision: first 6 layers | fusion: last 6 layers | multi_modal: all 12 layers 
        # )
        # start_logits, end_logits = of_logits.split(1, dim=-1)
        # # 0.1 sample image by OF indices
        # pixel_shape = list(pixel_values.size())
        # pixel_shape[1] = nframe
        # sampled_pixel_values = torch.zeros(pixel_shape, device=pixel_values.device)
        # cand_start_index = []
        # cand_end_index = []
        # top_k = 2
        # for _ in range(top_k):
        #     logits = torch.cat([start_logits, end_logits], dim=0)
        #     prob = F.gumbel_softmax(logits, tau=0.5, dim=1)
        #     # lb = torch.zeros(prob.shape).to(prob.device)
        #     # prob = torch.where(prob<0.85, lb, prob)
        #     index = prob.argmax(dim=1)
        #     cand_start_index.append(index[:batch_size])
        #     cand_end_index.append(index[batch_size:])
        pixel_shape = list(pixel_values.size())
        pixel_shape[1] = nframe
        sampled_pixel_values = torch.zeros(pixel_shape, device=pixel_values.device)
        video_lengths = batch["of_lengths"]
        # get frame idx
        for j in range(len(video_lengths)):
            video_length = video_lengths[j]
            # video_length = num_frames + 2
            
            # # sample for LSTP
            # cand_index = set()
            # # gumbel softmax
            # for ii in range(len(cand_start_index)):
            #     cand_start = cand_start_index[ii][j]
            #     cand_end = cand_end_index[ii][j]
            #     if cand_start >= video_length or cand_end >= video_length or (cand_start == 0 and cand_end == 0):
            #         cand_start = 0
            #         cand_end = video_length - 1
            #     # if cand_end < cand_start: cand_end = cand_start
            #     start, end = int(cand_start/video_length*num_frames), int(cand_end/video_length*num_frames)
            #     cand_index = set.union(cand_index, set(list(range(start, end))))
            # cand_index = sorted(list(cand_index))
            
            # uniform for baseline
            cand_index = list(range(num_frames))
            
            if cand_index == []:
                cand_index = list(range(num_frames))
            while len(cand_index) < nframe:
                cand_index = [xx for x in cand_index for xx in (x, x)]
            if len(cand_index) > nframe:
                intv = np.linspace(start=0, stop=len(cand_index), num=nframe+1).astype(int)
                new_cand_index = [cand_index[(intv[x]+intv[x+1]-1)//2] for x in range(len(intv)-1)]
                # new_cand_index = [cand_index[random.randrange(intv[x],intv[x+1])] for x in range(len(intv)-1)]
                cand_index = new_cand_index
            assert len(cand_index) == nframe
            cand_index = torch.tensor(cand_index, device=pixel_values.device)
            sampled_pixel_values[j] = torch.index_select(pixel_values[j], 0, cand_index) 
        
        # 1. get image embedding from vision encoder
        sampled_pixel_values = sampled_pixel_values.view(batch_size*nframe, sampled_pixel_values.size(2), sampled_pixel_values.size(3), sampled_pixel_values.size(4))
        image_embeddings = self.model.vision_model(
            pixel_values=sampled_pixel_values,
            return_dict=True
        ).last_hidden_state
        image_attention_mask = torch.ones(image_embeddings.size()[:-1], dtype=torch.long, device=batch["frames"].device)
        
        # 2. get qformer embedding
        image_attention_masks = torch.ones(image_embeddings.size()[:-1], dtype=torch.long, device=image_embeddings.device)
        query_tokens = self.model.query_tokens.expand(image_embeddings.shape[0], -1, -1)
        query_output = self.model.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeddings,
            encoder_attention_mask=image_attention_masks
        )[0]

        # 3. generate conditioned on query_output
        ## reshape qformer representation
        language_model_inputs = self.model.language_projection(query_output)
        language_model_inputs = language_model_inputs.reshape(batch_size, -1, language_model_inputs.shape[-1])
        language_attention_mask = torch.ones(
            language_model_inputs.size()[:-1], dtype=torch.long, device=language_model_inputs.device
        )
        # ## concatenate qformer representation and OF feature
        # of_feat = self.model.temporal_projection(of_feat)
        # of_feat_attn = torch.ones(
        #     of_feat.size()[:-1], dtype=torch.long, device=of_feat.device
        # )
        ## LLM
        # attention_mask = torch.cat([of_feat_attn, language_attention_mask, batch["question_attention_mask"]], dim=1)
        # inputs_embeddings = self.model.get_input_embeddings()(batch["question"])
        # inputs_embeddings = torch.cat([of_feat, language_model_inputs, inputs_embeddings], dim=1)
        
        attention_mask = torch.cat([language_attention_mask, batch["question_attention_mask"]], dim=1)
        inputs_embeddings = self.model.get_input_embeddings()(batch["question"])
        inputs_embeddings = torch.cat([language_model_inputs, inputs_embeddings], dim=1)
        
        generate_args = self.generate_configs
        # print(generate_args)
        outputs = self.model.language_model.generate(
            inputs_embeds=inputs_embeddings,
            attention_mask=attention_mask,
            # length_penalty=1,
            # repetition_penalty=1.5,
            # num_beams=5,
            # min_length=1,
            # max_length=250,
            # top_p=0.9,
            # temperature=0.8
            **generate_args
        )
        if self.model.config.text_config.architectures[0] == "LLaMAForCausalLM":
            outputs[outputs==0] = 2
            
        return outputs

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        # self.val_rouge_score.reset()
        self.val_bleu_score.reset()
        self.val_score_best.reset()

    def model_step(
        self, batch: Dict[str, torch.Tensor]
    ):
        
        loss, logits = self.forward(batch)
        labels = batch["answer"]
        return loss, logits, labels
    
    def eval_model_step(
        self, batch
    ):
        outputs = self.eval_forward(batch)
        labels = batch["text_answer"]
        predict = self.processor.batch_decode(outputs, skip_special_tokens=True)
        return predict, labels

    def training_step(
        self, batch, batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True)
        # self.log("train/ppl_socre", self.train_score, on_step=False, on_epoch=True, prog_bar=True)

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
        
        preds, targets = self.eval_model_step(batch)
        
        # print(f'preds {preds}')
        # print(f'target {targets}')
        if batch_idx % 100 == 0:
            print("=============================================")
            print(preds)
            print(targets)
            print("=============================================")

        # update and log metrics
        # self.val_rouge_score(preds, targets) # rouge1_fmeasure, rouge1_precision, rouge1_recall, rouge2_fmeasure, rouge2_precision, rouge2_recall, rougeL_fmeasure, rougeL_precision, rougeL_recall, rougeLsum_fmeasure, rougeLsum_precision, rougeLsum_recall
        self.val_bleu_score(preds, targets)
        # self.val_bert_score(preds, targets) # f1, precision, recall
        # self.log("val/rouge_score", self.val_rouge_score, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/score", self.val_bleu_score, on_step=False, on_epoch=True, prog_bar=True)
        # self.log("val/bert_score", self.val_bert_score, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        # score = self.val_bert_score.compute()['f1']  # get current val score
        score = self.val_bleu_score.compute()
        self.val_score_best(score)  # update best so far val score
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/score_best", self.val_score_best.compute(), sync_dist=True, prog_bar=True)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        
        preds, targets = self.eval_model_step(batch)

        # update and log metrics
        # self.test_rouge_score(preds, targets)
        self.test_bleu_score(preds, targets)
        # self.test_bert_score(preds, targets)
        # self.log("test/rouge_score", self.test_rouge_score, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/bleu_score", self.test_bleu_score, on_step=False, on_epoch=True, prog_bar=True)
        # self.log("test/bert_score", self.test_bert_score, on_step=False, on_epoch=True, prog_bar=True)

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
        # # original scheduler
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
        for param in self.of_extractor.parameters():
            param.requires_grad = False
        for param in self.model.vision_model.parameters():
            param.requires_grad = False
        for param in self.model.language_model.parameters():
            param.requires_grad = False
            

if __name__ == "__main__":
    _ = LSTPModule(None, None, None)
