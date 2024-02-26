from typing import Any, Dict, Tuple

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
from transformers import InstructBlipProcessor, BertConfig, AutoProcessor
from sentence_transformers import SentenceTransformer, util

from .components.xinstructblip import InstructBlipForConditionalGeneration
from .components.xropebert import RopeBertModel
from .components.xsampler import TemporalOFEmbedding
from .components.xraft import RAFT, InputPadder
from .components.raft_utils.utils import dp_state_to_normal
from src.gadgets.my_metrics import rouge_n

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
        self.model = InstructBlipForConditionalGeneration.from_pretrained(model_name_or_path)
        self.processor = AutoProcessor.from_pretrained(model_name_or_path, truncation_side="left")
        # if self.model.config.pad_token_id == -1:
        #     self.model.config.pad_token_id = self.processor.tokenizer.pad_token_id
        config = BertConfig(fusion_layer=6, encoder_width=768)
        self.temporal_encoder = RopeBertModel.from_pretrained(sampler_name_or_path, config=config)
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
        self.train_mrc_loss = MeanMetric()
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

        # # OF encoder
        # of = []
        # for frames in pixel_values:
        #     padder = InputPadder(frames[0].shape)
        #     frames = padder.pad(frames)
        #     frames_feats = self.of_extractor(frames[:-1], frames[1:])
        #     frames_feats = torch.cat([frames_feats, frames_feats[-1].unsqueeze(0)], dim=0) # repeat last of
        #     of.append(frames_feats)
        # of = torch.stack(of)
        # of_mask = torch.ones(of.size(0), of.size(1)+2, dtype=torch.long, device=of.device)

        # upload flow
        of = batch["of"]
        of_mask = batch["of_mask"]
        # # # upload flow_rgb
        # of = batch["of_rgb"]
        # of_mask = batch["of_rgb_mask"]

        # self-refinement
        # 1) get pseudo labels
        with torch.no_grad():
            # TODO: use loop to prevent from OOM
            image_embeddings = self.model.vision_model(
                pixel_values=batch["frames"],
                return_dict=True
            ).last_hidden_state
            image_attention_mask = torch.ones(image_embeddings.size()[:-1], dtype=torch.long, device=batch["frames"].device)
            
            # # i) per batch
            predict = []
            for ii in range(batch_size):
                batch_image_embeddings = image_embeddings[ii*num_frames: (ii+1)*num_frames]
                batch_image_attention_mask = image_attention_mask[ii*num_frames: (ii+1)*num_frames]
                # # ii) we take the nframe as minibatch: ensure num_frames is divided by nframe
                minibatch = num_frames // nframe
                for jj in range(minibatch):
                    minibatch_image_embeddings = batch_image_embeddings[jj*nframe: (jj+1)*nframe]
                    minibatch_image_attention_mask = batch_image_attention_mask[jj*nframe: (jj+1)*nframe]
                    query_tokens = self.model.query_tokens.expand(nframe, -1, -1)
                    query_attention_mask = torch.ones(query_tokens.size()[:-1], dtype=torch.long, device=batch["frames"].device)
                    qformer_text_inputs = torch.repeat_interleave(batch["qformer_text"][ii].unsqueeze(0), nframe, 0)
                    qformer_text_attention_mask = torch.repeat_interleave(batch["qformer_text_attention_mask"][ii].unsqueeze(0), nframe, 0)
                    qformer_attention_mask = torch.cat([query_attention_mask, qformer_text_attention_mask], dim=1)
                    query_outputs = self.model.qformer(
                        input_ids=qformer_text_inputs,
                        attention_mask=qformer_attention_mask,
                        query_embeds=query_tokens,
                        encoder_hidden_states=minibatch_image_embeddings,
                        encoder_attention_mask=minibatch_image_attention_mask,
                        return_dict=True,
                    )
                    query_output = query_outputs.last_hidden_state[:, :query_tokens.size(1), :]
                    
                    language_model_inputs = self.model.language_projection(query_output)
                    language_attention_mask = torch.ones(
                        language_model_inputs.size()[:-1], dtype=torch.long, device=language_model_inputs.device
                    )
                    question_attention_mask = torch.repeat_interleave(batch["question_attention_mask"][ii].unsqueeze(0), nframe, 0)
                    question = torch.repeat_interleave(batch["question"][ii].unsqueeze(0), nframe, 0)
                    attention_mask = torch.cat([language_attention_mask, question_attention_mask], dim=1)
                    inputs_embeddings = self.model.get_input_embeddings()(question)
                    # attention_mask = torch.cat([language_attention_mask, batch["question_attention_mask"]], dim=1)
                    # inputs_embeddings = self.model.get_input_embeddings()(batch["question"])
                    inputs_embeddings = torch.cat([language_model_inputs, inputs_embeddings], dim=1)                    
                    outputs = self.model.language_model.generate(
                        inputs_embeds=inputs_embeddings,
                        attention_mask=attention_mask,
                        max_length=128,
                    )
                    if self.model.config.text_config.architectures[0] == "LLaMAForCausalLM":
                        outputs[outputs==0] = 2
                    predict.extend(self.processor.batch_decode(outputs, skip_special_tokens=True))
            
            # query_tokens = self.model.query_tokens.expand(image_embeddings.shape[0], -1, -1)
            # query_attention_mask = torch.ones(query_tokens.size()[:-1], dtype=torch.long, device=batch["frames"].device)
            # # qformer_attention_mask = torch.cat([query_attention_mask, batch["qformer_text_attention_mask"]], dim=1)
            # qformer_text_inputs = torch.repeat_interleave(batch["qformer_text"], image_embeddings.shape[0]//batch_size, 0)
            # qformer_text_attention_mask = torch.repeat_interleave(batch["qformer_text_attention_mask"], image_embeddings.shape[0]//batch_size, 0)
            # qformer_attention_mask = torch.cat([query_attention_mask, qformer_text_attention_mask], dim=1)
            # query_outputs = self.model.qformer(
            #     input_ids=qformer_text_inputs,
            #     attention_mask=qformer_attention_mask,
            #     query_embeds=query_tokens,
            #     encoder_hidden_states=image_embeddings,
            #     encoder_attention_mask=image_attention_mask,
            #     return_dict=True,
            # )
            # query_output = query_outputs.last_hidden_state[:, :query_tokens.size(1), :]
            
            # language_model_inputs = self.model.language_projection(query_output)
            # language_attention_mask = torch.ones(
            #     language_model_inputs.size()[:-1], dtype=torch.long, device=language_model_inputs.device
            # )
            # question_attention_mask = torch.repeat_interleave(batch["question_attention_mask"], query_output.shape[0]//batch_size, 0)
            # question = torch.repeat_interleave(batch["question"], query_output.shape[0]//batch_size, 0)
            # attention_mask = torch.cat([language_attention_mask, question_attention_mask], dim=1)
            # inputs_embeddings = self.model.get_input_embeddings()(question)
            # # attention_mask = torch.cat([language_attention_mask, batch["question_attention_mask"]], dim=1)
            # # inputs_embeddings = self.model.get_input_embeddings()(batch["question"])
            # inputs_embeddings = torch.cat([language_model_inputs, inputs_embeddings], dim=1)
            
            # outputs = self.model.language_model.generate(
            #     inputs_embeds=inputs_embeddings,
            #     attention_mask=attention_mask,
            #     max_length=128,
            # )
            # if self.model.config.text_config.architectures[0] == "LLaMAForCausalLM":
            #     outputs[outputs==0] = 2
            # predict = self.processor.batch_decode(outputs, skip_special_tokens=True)
            
            ## evaluate results -> pseudo labels: better bleu score means better understand of the results
            target = batch["text_answer"]
            target = [target[int(idx//num_frames)] for idx in range(len(predict))]
            scores = rouge_n(target, predict)
            scores = torch.tensor(scores, dtype=torch.float)
            scores = scores.view(batch_size, num_frames)
            # monotone stack -> pseudo span
            start_targets = []
            end_targets = []
            for score in scores:
                bs = 0
                start_target = 0
                end_target = len(score) - 1
                stack = []
                score = [0] + score.tolist() + [0]
                for i in range(len(score)):
                    while stack and score[stack[-1]] > score[i]:
                        tmp = stack.pop()
                        tmp_bs = (i-stack[-1]-1) * score[tmp]
                        if tmp_bs > bs:
                            bs = tmp_bs
                            start_target, end_target = stack[-1], i-2
                    stack.append(i)
                start_targets.append(start_target)
                end_targets.append(end_target)

            flow_lengths = batch["of_lengths"]
            start_targets = [int(start_targets[ii] / (num_frames-1) * (flow_lengths[ii]-1)) for ii in range(batch_size)]
            end_targets = [int(end_targets[ii] / (num_frames-1) * (flow_lengths[ii]-1)) for ii in range(batch_size)]
            
            start_targets = torch.tensor(start_targets, dtype=torch.long, device=pixel_values.device)
            end_targets = torch.tensor(end_targets, dtype=torch.long, device=pixel_values.device)

            # select_frames_idx = torch.topk(scores, nframe, dim=-1).indices.tolist()
            # refine_pos = []
            # for pos in select_frames_idx:
            #     refine_pos.append(sorted(pos))
        # 2) optimize temporal encoder
        of_feat, of_logits = self.temporal_encoder(
            encoder_embeds=of,
            attention_mask=of_mask,
            encoder_hidden_states=batch["sampler_question"],
            encoder_attention_mask=batch["sampler_question_attention_mask"],
            mode="fusion" # text/vision, fusion, multi_modal
        )
        start_logits, end_logits = of_logits.split(1, dim=-1)
        ignored_index = start_logits.size(1)
        # loss_fct = CrossEntropyLoss(reduction="mean")
        loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
        if len(start_targets.size()) > 1:
            start_targets = start_targets.squeeze(-1)
        if len(end_targets.size()) > 1:
            end_targets = end_targets.squeeze(-1)
        start_targets = start_targets.clamp(0, ignored_index)
        end_targets = end_targets.clamp(0, ignored_index)
        start_loss = loss_fct(start_logits.squeeze(-1).contiguous(), start_targets)
        end_loss = loss_fct(end_logits.squeeze(-1).contiguous(), end_targets)
        mrc_loss = (start_loss + end_loss) / 2
        
        # 0. get OF embedding from temporal encoder
        # of_feat, of_logits = self.temporal_encoder(
        #     encoder_embeds=of,
        #     attention_mask=of_mask,
        #     # encoder_embeds=batch["of"],
        #     # attention_mask=batch["of_mask"],
        #     encoder_hidden_states=batch["sampler_question"],
        #     encoder_attention_mask=batch["sampler_question_attention_mask"],
        #     mode="multi_modal" # text/vision: first 6 layers | fusion: last 6 layers | multi_modal: all 12 layers 
        # )
        # start_logits, end_logits = of_logits.split(1, dim=-1)
        # 0.1 sample image by OF indices
        pixel_shape = list(pixel_values.size())
        pixel_shape[1] = nframe
        sampled_pixel_values = torch.zeros(pixel_shape, device=pixel_values.device)
        cand_start_index = []
        cand_end_index = []
        top_k = 2
        for _ in range(top_k):
            logits = torch.cat([start_logits, end_logits], dim=0)
            prob = F.gumbel_softmax(logits, tau=0.5, dim=1)
            # lb = torch.zeros(prob.shape).to(prob.device)
            # prob = torch.where(prob<0.85, lb, prob)
            index = prob.argmax(dim=1)
            cand_start_index.append(index[:batch_size])
            cand_end_index.append(index[batch_size:])
        video_lengths = batch["of_lengths"]
        
        # get frame idx
        for j in range(len(video_lengths)):
            video_length = video_lengths[j]
            # video_length = num_frames + 2
            cand_index = set()
            # gumbel softmax
            for ii in range(len(cand_start_index)):
                cand_start = cand_start_index[ii][j]
                cand_end = cand_end_index[ii][j]
                if cand_start >= video_length or cand_end >= video_length or (cand_start == 0 and cand_end == 0):
                    cand_start = 0
                    cand_end = video_length - 1
                # if cand_end < cand_start: cand_end = cand_start
                start, end = int(cand_start*(num_frames-1)/(video_length-1)), int(cand_end*(num_frames-1)/(video_length-1))
                cand_index = set.union(cand_index, set(list(range(start, end))))
            cand_index = sorted(list(cand_index))
            
            # # uniform for baseline
            # cand_index = list(range(num_frames))
            
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
        query_attention_mask = torch.ones(query_tokens.size()[:-1], dtype=torch.long, device=image_embeddings.device)
        # qformer_attention_mask = batch["qformer_text_attention_mask"]
        qformer_text_inputs = torch.repeat_interleave(batch["qformer_text"], nframe, 0)
        qformer_text_attention_mask = torch.repeat_interleave(batch["qformer_text_attention_mask"], nframe, 0)
        qformer_attention_mask = torch.cat([query_attention_mask, qformer_text_attention_mask], dim=1)
        query_outputs = self.model.qformer(
            input_ids=qformer_text_inputs,
            attention_mask=qformer_attention_mask,
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeddings,
            encoder_attention_mask=image_attention_masks,
        )
        query_output = query_outputs[0][:, :query_tokens.size(1), :]
        
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
        # inputs_embeddings = self.model.language_model.get_input_embeddings()(batch["texts"])
        # inputs_embeddings = torch.cat([language_model_inputs, inputs_embeddings.to(language_model_inputs.device)], dim=1)
        # attention_mask = torch.cat([language_model_attention_mask.to(batch["text_attention_mask"]), batch["text_attention_mask"]], dim=1)
        if self.model.config.use_decoder_only_language_model:
            llm_tokens, input_part_targets_len = self.concat_text_input_output(
                batch["question"],
                batch["question_attention_mask"],
                batch["answer"],
                batch["answer_attention_mask"]
            )
            labels = llm_tokens["input_ids"].masked_fill(llm_tokens["input_ids"] == self.processor.tokenizer.pad_token_id, -100)
            for i, l in enumerate(input_part_targets_len):
                labels[i][:l] = -100
            empty_labels = (torch.ones(language_model_attention_mask.size(), dtype=torch.long).to(language_model_attention_mask.device).fill_(-100))
            labels = torch.cat([empty_labels, labels], dim=1)
            inputs_embeddings = self.model.language_model.get_input_embeddings()(llm_tokens["input_ids"])
            inputs_embeddings = torch.cat([language_model_inputs, inputs_embeddings], dim=1)
            attention_mask = torch.cat([language_model_attention_mask, llm_tokens["attention_mask"]], dim=1)
            outputs = self.model.language_model(
                inputs_embeds=inputs_embeddings,
                attention_mask=attention_mask
            )
            logits = outputs[0]
            loss = None
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous().to(logits.device)
            # inputs_embeddings = self.model.language_model.get_input_embeddings()(batch["instruction"])
            # # inputs_embeddings = torch.cat([of_feat, language_model_inputs, inputs_embeddings.to(language_model_inputs.device)], dim=1)
            # # attention_mask = torch.cat([of_feat_attn, language_model_attention_mask.to(batch["instruction_attention_mask"]), batch["instruction_attention_mask"]], dim=1)
            # inputs_embeddings = torch.cat([language_model_inputs, inputs_embeddings.to(language_model_inputs.device)], dim=1)
            # attention_mask = torch.cat([language_model_attention_mask.to(batch["instruction_attention_mask"]), batch["instruction_attention_mask"]], dim=1)
            # outputs = self.model.language_model(
            #     inputs_embeds=inputs_embeddings,
            #     attention_mask=attention_mask
            # )
            # logits = outputs[0]
            # loss = None
            # labels = batch["instruction"]
            # logits = logits[:, -labels.size(1):, :]
            # shift_logits = logits[..., :-1, :].contiguous()
            # shift_labels = labels[..., 1:].contiguous().to(logits.device)
            loss_fct = CrossEntropyLoss(reduction="mean")
            loss = loss_fct(shift_logits.view(-1, self.model.config.text_config.vocab_size), shift_labels.view(-1))
        else:
            inputs_embeddings = self.model.language_model.get_input_embeddings()(batch["question"])
            # inputs_embeddings = torch.cat([of_feat, language_model_inputs, inputs_embeddings.to(language_model_inputs.device)], dim=1)
            # attention_mask = torch.cat([of_feat_attn, language_model_attention_mask.to(batch["question_attention_mask"]), batch["question_attention_mask"]], dim=1)
            inputs_embeddings = torch.cat([language_model_inputs, inputs_embeddings.to(language_model_inputs.device)], dim=1)
            attention_mask = torch.cat([language_model_attention_mask.to(batch["question_attention_mask"]), batch["question_attention_mask"]], dim=1)
            labels = batch["answer"].masked_fill(batch["answer"] == self.processor.tokenizer.pad_token_id, -100)
            outputs = self.model.language_model(
                inputs_embeds=inputs_embeddings,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs[0]
            logits = outputs[1]
        
        return loss, logits, mrc_loss
    
    @torch.no_grad()
    def eval_forward(self, batch):
        batch_size = batch["answer"].shape[0]
        nframe = batch["nframe"]
        pixel_values = batch['frames']
        num_frames = pixel_values.size(0) // batch_size
        pixel_values = pixel_values.view(batch_size, num_frames, pixel_values.size(1), pixel_values.size(2), pixel_values.size(3))
        
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
        
        # upload flow
        of = batch["of"]
        of_mask = batch["of_mask"]
        # # # upload flow_rgb
        # of = batch["of_rgb"]
        # of_mask = batch["of_rgb_mask"]
        
        of_feat, of_logits = self.temporal_encoder(
            encoder_embeds=of,
            attention_mask=of_mask,
            encoder_hidden_states=batch["sampler_question"],
            encoder_attention_mask=batch["sampler_question_attention_mask"],
            mode="fusion" # text/vision: first 6 layers | fusion: last 6 layers | multi_modal: all 12 layers 
        )
        start_logits, end_logits = of_logits.split(1, dim=-1)
        # 0.1 sample image by OF indices
        pixel_shape = list(pixel_values.size())
        pixel_shape[1] = nframe
        sampled_pixel_values = torch.zeros(pixel_shape, device=pixel_values.device)
        cand_start_index = []
        cand_end_index = []
        top_k = 2
        for _ in range(top_k):
            logits = torch.cat([start_logits, end_logits], dim=0)
            prob = F.gumbel_softmax(logits, tau=0.5, dim=1)
            # lb = torch.zeros(prob.shape).to(prob.device)
            # prob = torch.where(prob<0.85, lb, prob)
            index = prob.argmax(dim=1)
            cand_start_index.append(index[:batch_size])
            cand_end_index.append(index[batch_size:])
        video_lengths = batch["of_lengths"]
        # get frame idx
        for j in range(len(video_lengths)):
            video_length = video_lengths[j]
            # video_length = num_frames + 2
            cand_index = set()
            # gumbel softmax
            for ii in range(len(cand_start_index)):
                cand_start = cand_start_index[ii][j]
                cand_end = cand_end_index[ii][j]
                if cand_start >= video_length or cand_end >= video_length or (cand_start == 0 and cand_end == 0):
                    cand_start = 0
                    cand_end = video_length - 1
                # if cand_end < cand_start: cand_end = cand_start
                start, end = int(cand_start*(num_frames-1)/(video_length-1)), int(cand_end*(num_frames-1)/(video_length-1))
                cand_index = set.union(cand_index, set(list(range(start, end))))
            cand_index = sorted(list(cand_index))
            # # uniform for baseline
            # cand_index = list(range(num_frames))
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
        query_tokens = self.model.query_tokens.expand(image_embeddings.shape[0], -1, -1)
        query_attention_mask = torch.ones(query_tokens.size()[:-1], dtype=torch.long, device=batch["frames"].device)
        qformer_text_inputs = torch.repeat_interleave(batch["qformer_text"], nframe, 0)
        qformer_text_attention_mask = torch.repeat_interleave(batch["qformer_text_attention_mask"], nframe, 0)
        qformer_attention_mask = torch.cat([query_attention_mask, qformer_text_attention_mask], dim=1)
        query_outputs = self.model.qformer(
            input_ids=qformer_text_inputs,
            attention_mask=qformer_attention_mask,
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeddings,
            encoder_attention_mask=image_attention_mask,
            return_dict=True,
        )
        query_output = query_outputs.last_hidden_state[:, :query_tokens.size(1), :]
        
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
        
        # generate_args = self.hparams["generate_configs"]
        generate_args = self.generate_configs
        outputs = self.model.language_model.generate(
            inputs_embeds=inputs_embeddings,
            attention_mask=attention_mask,
            **generate_args,
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
        
        loss, logits, mrc_loss = self.forward(batch)
        labels = batch["answer"]
        return loss, logits, labels, mrc_loss
    
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
        loss, preds, targets, mrc_loss = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True)
        # self.log("train/ppl_socre", self.train_score, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/mrc_loss", self.train_mrc_loss, on_step=True, on_epoch=True, prog_bar=True)

        # return loss or backpropagation will fail
        return loss + mrc_loss

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

        # return {"optimizer": optimizer}

        
    def freeze_weights(self):
        for param in self.model.vision_model.parameters():
            param.requires_grad = False
        for param in self.model.language_model.parameters():
            param.requires_grad = False

    def dp_state_to_normal(state_dict):
        '''Converts a torch.DataParallel checkpoint to regular'''
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module'):
                new_state_dict[k.replace('module.', '')] = v
        return new_state_dict
            
    def concat_text_input_output(self, input_ids, input_atts, output_ids, output_atts):
        input_part_targets_len = []
        llm_tokens = {"input_ids": [], "attention_mask": []}
        for i in range(input_ids.size(0)):
            this_input_ones = input_atts[i].sum()
            input_part_targets_len.append(this_input_ones)
            llm_tokens['input_ids'].append(
                torch.cat([
                    input_ids[i][:this_input_ones],
                    output_ids[i][1:],
                    input_ids[i][this_input_ones:]
                ])
            )
            llm_tokens['attention_mask'].append(
                torch.cat([
                    input_atts[i][:this_input_ones],
                    output_atts[i][1:],
                    input_atts[i][this_input_ones:]
                ])
            )
        llm_tokens['input_ids'] = torch.stack(llm_tokens['input_ids'])
        llm_tokens['attention_mask'] = torch.stack(llm_tokens['attention_mask'])
        return llm_tokens, input_part_targets_len

if __name__ == "__main__":
    _ = LSTPSFModule(None, None, None)
