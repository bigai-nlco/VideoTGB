from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import InstructBlipProcessor, BertConfig, InstructBlipConfig, Blip2Config
from sentence_transformers import SentenceTransformer, util

from src.models.components.xinstructblip import InstructBlipForConditionalGeneration
from src.models.components.xblip2 import Blip2ForConditionalGeneration
from src.models.components.xropebert import RopeBertModel
from src.models.components.xsampler import TemporalOFEmbedding
from src.models.components.xraft import RAFT, InputPadder
from src.models.components.raft_utils.utils import dp_state_to_normal

class LSTP(nn.Module):
    
    def __init__(
        self,
        base_model_path,
        device
    ) -> None:
        super().__init__()
        
        # hyperparameter
        # self.generate_configs = generate_configs

        # load model and init weight
        model_config = InstructBlipConfig.from_pretrained(base_model_path)
        self.model = InstructBlipForConditionalGeneration(config=model_config)
        temporal_encoder_config = BertConfig(fusion_layer=6, encoder_width=768)
        self.temporal_encoder = RopeBertModel(config=temporal_encoder_config)
        self.of_extractor = RAFT()
        self.device = device
        
    @torch.no_grad()
    def generate(
        self,
        frames,
        nframe,
        text_encoding,
        sampler_text_encoding,
        do_sample=True,
        # max_new_tokens=256,
        # temperature=0.8,
        sampling="Beam search",
        min_len=1,
        max_len=128,
        num_beams=5,
        top_p=0.9,
        length_penalty=1.0,
        repetition_penalty=1.5,
    ):
        batch_size = sampler_text_encoding.input_ids.shape[0]
        pixel_values = frames
        num_frames = pixel_values.size(0) // batch_size
        pixel_values = pixel_values.view(batch_size, num_frames, pixel_values.size(1), pixel_values.size(2), pixel_values.size(3))
        
        # 0. get OF embeeding from temporal encoder
        # 0.0 get OF feature from RAFT
        of = []
        for frames in pixel_values:
            padder = InputPadder(frames[0].shape)
            frames = padder.pad(frames)
            frames_feats = self.of_extractor(frames[:-1], frames[1:])
            frames_feats = torch.cat([frames_feats, frames_feats[-1].unsqueeze(0)], dim=0) # repeat last of
            of.append(frames_feats)
        of = torch.stack(of)
        of_mask = torch.ones(of.size(0), of.size(1)+2, dtype=torch.long, device=of.device)
        # print("===========================================")
        # print(of.size())
        # print(batch["of"].size())
        # print(of_mask.size())
        # print(batch["of_mask"].size())
        # print("===========================================")
        of_feat, of_logits = self.temporal_encoder(
            encoder_embeds=of,
            attention_mask=of_mask,
            # encoder_embeds=batch["of"],
            # attention_mask=batch["of_mask"],
            encoder_hidden_states=sampler_text_encoding["input_ids"],
            encoder_attention_mask=sampler_text_encoding["attention_mask"],
            mode="multi_modal" # text/vision: first 6 layers | fusion: last 6 layers | multi_modal: all 12 layers 
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
        video_lengths = [frames.size(1)]
        # get frame idx
        for j in range(len(video_lengths)):
            video_length = video_lengths[j]
            cand_index = set()
            # gumbel softmax
            for ii in range(len(cand_start_index)):
                cand_start = cand_start_index[ii][j]
                cand_end = cand_end_index[ii][j]
                if cand_start >= video_length or cand_end >= video_length or (cand_start == 0 and cand_end == 0):
                    cand_start = 0
                    cand_end = video_length - 1
                # if cand_end < cand_start: cand_end = cand_start
                start, end = int(cand_start/video_length*num_frames), int(cand_end/video_length*num_frames)
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
        frames_feats = torch.index_select(frames_feats, 0, cand_index)
        
        # 1. get image embedding from vision encoder
        sampled_pixel_values = sampled_pixel_values.view(batch_size*nframe, sampled_pixel_values.size(2), sampled_pixel_values.size(3), sampled_pixel_values.size(4))
        image_embeddings = self.model.vision_model(
            pixel_values=sampled_pixel_values,
            return_dict=True
        ).last_hidden_state
        image_attention_mask = torch.ones(image_embeddings.size()[:-1], dtype=torch.long, device=frames.device)
        
        # 2. get qformer embedding
        query_tokens = self.model.query_tokens.expand(image_embeddings.shape[0], -1, -1)
        query_attention_mask = torch.ones(query_tokens.size()[:-1], dtype=torch.long, device=frames.device)
        qformer_text_inputs = torch.repeat_interleave(text_encoding["qformer_input_ids"], nframe, 0)
        qformer_text_attention_mask = torch.repeat_interleave(text_encoding["qformer_attention_mask"], nframe, 0)
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
        ## concatenate qformer representation and OF feature
        # of_feat = self.model.temporal_projection(of_feat)
        # of_feat_attn = torch.ones(
        #     of_feat.size()[:-1], dtype=torch.long, device=of_feat.device
        # )
        # attention_mask = torch.cat([of_feat_attn, language_attention_mask, text_encoding["attention_mask"]], dim=1)
        # inputs_embeddings = self.model.get_input_embeddings()(text_encoding["input_ids"])
        # inputs_embeddings = torch.cat([of_feat, language_model_inputs, inputs_embeddings], dim=1)
        attention_mask = torch.cat([language_attention_mask, text_encoding["attention_mask"]], dim=1)
        inputs_embeddings = self.model.get_input_embeddings()(text_encoding["input_ids"])
        inputs_embeddings = torch.cat([language_model_inputs, inputs_embeddings], dim=1)
        

        outputs = self.model.language_model.generate(
            inputs_embeds=inputs_embeddings,
            attention_mask=attention_mask,
            do_sample=do_sample,
            # temperature=temperature,
            # max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            min_length=min_len,
            max_length=max_len,
            top_p=top_p,
            length_penalty=length_penalty,
            repetition_penalty=repetition_penalty,
        )
        if self.model.config.text_config.architectures[0] == "LLaMAForCausalLM":
            outputs[outputs==0] = 2

        return outputs, cand_index, frames_feats


class LSTP_blip2(nn.Module):
    
    def __init__(
        self,
        base_model_path,
        device
    ) -> None:
        super().__init__()
        
        # hyperparameter
        # self.generate_configs = generate_configs

        # load model and init weight
        model_config = Blip2Config.from_pretrained(base_model_path)
        self.model = Blip2ForConditionalGeneration(config=model_config)
        temporal_encoder_config = BertConfig(fusion_layer=6, encoder_width=768)
        self.temporal_encoder = RopeBertModel(config=temporal_encoder_config)
        self.of_extractor = RAFT()
        self.device = device
        
    @torch.no_grad()
    def generate(
        self,
        frames,
        nframe,
        text_encoding,
        sampler_text_encoding,
        do_sample=True,
        # temperature=0.8,
        # max_new_tokens=256,
        # repetition_penalty=1.5,
        sampling="Beam search",
        min_len=1,
        max_len=128,
        num_beams=5,
        top_p=0.9,
        length_penalty=1.0,
        repetition_penalty=1.5,
    ):
        batch_size = sampler_text_encoding.input_ids.shape[0]
        pixel_values = frames
        num_frames = pixel_values.size(0) // batch_size
        pixel_values = pixel_values.view(batch_size, num_frames, pixel_values.size(1), pixel_values.size(2), pixel_values.size(3))
        
        # 0. get OF embeeding from temporal encoder
        # 0.0 get OF feature from RAFT
        of = []
        for frames in pixel_values:
            padder = InputPadder(frames[0].shape)
            frames = padder.pad(frames)
            frames_feats = self.of_extractor(frames[:-1], frames[1:])
            frames_feats = torch.cat([frames_feats, frames_feats[-1].unsqueeze(0)], dim=0) # repeat last of
            of.append(frames_feats)
        of = torch.stack(of)
        of_mask = torch.ones(of.size(0), of.size(1)+2, dtype=torch.long, device=of.device)
        # print("===========================================")
        # print(of.size())
        # print(batch["of"].size())
        # print(of_mask.size())
        # print(batch["of_mask"].size())
        # print("===========================================")
        of_feat, of_logits = self.temporal_encoder(
            encoder_embeds=of,
            attention_mask=of_mask,
            # encoder_embeds=batch["of"],
            # attention_mask=batch["of_mask"],
            encoder_hidden_states=sampler_text_encoding["input_ids"],
            encoder_attention_mask=sampler_text_encoding["attention_mask"],
            mode="multi_modal" # text/vision: first 6 layers | fusion: last 6 layers | multi_modal: all 12 layers 
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
        video_lengths = [frames.size(1)]
        # get frame idx
        for j in range(len(video_lengths)):
            # video_length = video_lengths[j]
            video_length = num_frames
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
        frames_feats = torch.index_select(frames_feats, 0, cand_index)
        
        # 1. get image embedding from vision encoder
        sampled_pixel_values = sampled_pixel_values.view(batch_size*nframe, sampled_pixel_values.size(2), sampled_pixel_values.size(3), sampled_pixel_values.size(4))
        image_embeddings = self.model.vision_model(
            pixel_values=sampled_pixel_values,
            return_dict=True
        ).last_hidden_state
        image_attention_mask = torch.ones(image_embeddings.size()[:-1], dtype=torch.long, device=frames.device)
        
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
        # language_model_inputs = torch.cat([of_feat, language_model_inputs], dim=1)
        # language_attention_mask = torch.cat([of_feat_attn, language_attention_mask], dim=1)
        ## LLM
        # attention_mask = torch.cat([of_feat_attn, language_attention_mask, text_encoding["attention_mask"]], dim=1)
        # inputs_embeddings = self.model.get_input_embeddings()(text_encoding["input_ids"])
        # inputs_embeddings = torch.cat([of_feat, language_model_inputs, inputs_embeddings], dim=1)
        attention_mask = torch.cat([language_attention_mask, text_encoding["attention_mask"]], dim=1)
        inputs_embeddings = self.model.get_input_embeddings()(text_encoding["input_ids"])
        inputs_embeddings = torch.cat([language_model_inputs, inputs_embeddings], dim=1)
        
        outputs = self.model.language_model.generate(
            inputs_embeds=inputs_embeddings,
            attention_mask=attention_mask,
            do_sample=do_sample,
            # temperature=temperature,
            # max_new_tokens=max_new_tokens,
            # repetition_penalty=repetition_penalty,
            num_beams=num_beams,
            min_length=min_len,
            max_length=max_len,
            top_p=top_p,
            length_penalty=length_penalty,
            repetition_penalty=repetition_penalty,
        )
        if self.model.config.text_config.architectures[0] == "LLaMAForCausalLM":
            outputs[outputs==0] = 2

        return outputs, cand_index, frames_feats
