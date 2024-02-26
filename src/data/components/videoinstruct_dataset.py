import os
import json
from PIL import Image
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, AutoProcessor, InstructBlipProcessor, BlipImageProcessor

from .util import read_videos, read_videos_cv2, read_videos_av, sample_frames, flow_to_image
from src.data.components import conversation as conversation_lib

class VIDEOINSTRUCT(Dataset):
    
    def __init__(self,
            text_dir: str,
            video_dir: str,
            of_dir: str,
            nframe: int,
            split: str,
            processor: Optional[Callable] = None,
            sampler_processor: Optional[Callable] = None,
            video_transform: Optional[Callable] = None,
            image_transform: Optional[Callable] = None,
        ):
        
        self.split = split
        self.video_transform = video_transform
        self.image_transform = image_transform
        self.processor = processor
        self.sampler_processor = sampler_processor
        
        self.video_dir = video_dir
        self.of_dir = of_dir
        self.text_dir = text_dir
        
        self.max_txt_len = 128
        self.nframe = nframe
        self.sampling = 'uniform'
        
        self.data = self._load_data()

        # pseudo_label
        data_path = os.path.join(self.text_dir,  'pseudo_label.json')
        with open(data_path) as jp:
            self.pseudo_label = json.load(jp)
        
    def __len__(self):
        
        return len(self.data)
    
    def __getitem__(self, index):
        
        data_dct = self.data[index]
        answer = data_dct["a"]
        question = data_dct["q"]
        question = "USER: " + question + "ASSISTANT: "
        answer = answer + " </s>"
        instruction = question + ' ' + answer
        vid = data_dct["video_id"]
        idx = data_dct['idx']
        
        # process video and of
        frames = self.get_frames(vid, self.video_dir) # TCHW
        of = self.get_of(vid, self.of_dir) # TCHW
        flows = []
        for flow in of:
            flow = flow_to_image(np.transpose(flow, (1, 2, 0))) # HWC
            flows.append(flow)
        of_rgb = torch.from_numpy(np.stack(flows, axis=0)) # THWC
        of_rgb = of_rgb.permute(3, 0, 1, 2).float() # CTHW
        of_rgb = self.video_transform(of_rgb)
        of_rgb = of_rgb.permute(1,0,2,3) # TCHW
        # of_length = of.size(0) + 2
        of_length = of.shape[0]
        of = self.normalize_flow(of)
        of = torch.from_numpy(of)

        # pseudo_label
        start = int(self.pseudo_label[idx][0] / 31 * (of_length-1))
        end = int(self.pseudo_label[idx][1] / 31 * (of_length-1))
        # of_length += 2

        return {'idx': idx, 'frames':frames, 'of':of, 'of_rgb':of_rgb, 'of_length':of_length, 'question':question, 'answer':answer, 'instruction':instruction, "idx": idx, "start": start, "end": end}
    
    def collate(self, batch):
        
        idxs = [x['idx'] for x in batch]

        # frames = [x['frames'] for x in batch]
        frames = torch.cat([data['frames'] for data in batch]) # B*T, 3, 224, 224
        answers = [x['answer'] for x in batch]
        questions = [x['question'] for x in batch]
        instruction = [x['instruction'] for x in batch]

        # preprocess image and flow
        of = pad_sequence([x['of'] for x in batch], batch_first=True)
        of_mask = torch.zeros(of.size(0), of.size(1)+2, dtype=torch.long)
        for i, data in enumerate(batch):
            of_mask[i, :data["of"].size(0)+2] = 1
        of_rgb = pad_sequence([x['of_rgb'] for x in batch], batch_first=True)
        of_rgb_mask = torch.zeros(of_rgb.size(0), of_rgb.size(1)+2, dtype=torch.long)
        for i, data in enumerate(batch):
            of_rgb_mask[i, :data["of_rgb"].size(0)+2] = 1
        of_lengths = [x['of_length'] for x in batch]
        starts = torch.tensor([x["start"] for x in batch], dtype=torch.long)
        ends = torch.tensor([x["end"] for x in batch], dtype=torch.long)

        # preprocess text
        sampler_question_encoding = self.sampler_processor(
            text=questions,
            padding='longest',
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        )
        if 'vicuna' in self.processor.tokenizer.name_or_path:
            self.processor.padding_side = "right"
            self.processor.truncation_side = "left"
        question_encoding = self.processor(
            text=questions,
            padding='longest',
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        )
        if 'vicuna' in self.processor.tokenizer.name_or_path:
            self.processor.truncation_side = "right" 
        answer_encoding = self.processor(
            text=answers,
            padding='longest',
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        )
        instruction_encoding = self.processor(
            text=instruction,
            padding='longest',
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        )
        
        
        if "instructblip" in self.processor.tokenizer.name_or_path:
            return {"idxs": idxs,
                    "frames": frames, 
                    "of": of,
                    "of_mask": of_mask,
                    "of_rgb": of_rgb,
                    "of_rgb_mask": of_rgb_mask,
                    "sampler_question": sampler_question_encoding["input_ids"],
                    "sampler_question_attention_mask": sampler_question_encoding["attention_mask"],
                    "question": question_encoding["input_ids"],
                    "question_attention_mask": question_encoding["attention_mask"],
                    "qformer_text":question_encoding["qformer_input_ids"],
                    "qformer_text_attention_mask": question_encoding["qformer_attention_mask"],
                    "instruction": instruction_encoding["input_ids"],
                    "instruction_attention_mask": instruction_encoding["attention_mask"],
                    "answer": answer_encoding["input_ids"],
                    "answer_attention_mask": answer_encoding["attention_mask"],
                    "text_answer": answers,
                    "nframe": self.nframe,
                    "of_lengths": of_lengths,
                    "starts": starts,
                    "ends": ends,
                    }
        elif "blip2" in self.processor.tokenizer.name_or_path:
            return {"idxs": idxs,
                    "frames": frames, 
                    "of": of,
                    "of_mask": of_mask,
                    "of_rgb": of_rgb,
                    "of_rgb_mask": of_rgb_mask,
                    "sampler_question": sampler_question_encoding["input_ids"],
                    "sampler_question_attention_mask": sampler_question_encoding["attention_mask"],
                    "question": question_encoding["input_ids"],
                    "question_attention_mask": question_encoding["attention_mask"],
                    # "qformer_text":question_encoding["qformer_input_ids"],
                    # "qformer_text_attention_mask": question_encoding["qformer_attention_mask"],
                    "instruction": instruction_encoding["input_ids"],
                    "instruction_attention_mask": instruction_encoding["attention_mask"],
                    "answer": answer_encoding["input_ids"],
                    "answer_attention_mask": answer_encoding["attention_mask"],
                    "text_answer": answers,
                    "nframe": self.nframe,
                    "of_lengths": of_lengths,
                    "starts": starts,
                    "ends": ends,
                    }
            
    
    def _load_data(self):
        
        data_path = os.path.join(self.text_dir, self.split + '.json')
        with open(data_path) as jp:
            data_dct = json.load(jp)
        data_lst = []
        for idx, dct in data_dct.items():
            dct['idx'] = idx
            data_lst.append(dct)
        return data_lst
            
    @staticmethod
    def rescale(x,max_range,min_range):
        max_val = np.max(x)
        min_val = np.min(x)
        return (max_range-min_range)/(max_val-min_val)*(x-max_val)+max_range
    @staticmethod
    def normalize_flow(flow):
        # N, 2, H, W -> N, H, W, 2
        # flow_uv = np.transpose(flow, (0, 2, 3, 1))
        flow_uv = flow.transpose(0,2,3,1)
        u = flow_uv[:,:,:,0]
        v = flow_uv[:,:,:,1]
        rad = np.sqrt(np.square(u) + np.square(v))
        rad_max = np.max(rad)
        epsilon = 1e-5
        u = u / (rad_max + epsilon)
        v = v / (rad_max + epsilon)
        normalized_flow_uv = np.stack([u,v], axis=-1)
        # normalized_flow =np.transpose(normalized_flow_uv, (0, 3, 1, 2))
        normalized_flow = normalized_flow_uv.transpose(0, 3, 1, 2)
        return normalized_flow

    def get_of(self, video_name, of_path):
        of_path = os.path.join(of_path, video_name+'_raft.npy')
        of = np.load(of_path) # num_of_frames, 2, H, W
        # of_npy = self.normalize_flow(of_npy) # not compatible with rgb
        # of = torch.tensor(of, dtype=torch.float)

        # # way1. cut off
        if of.shape[0] > 64:
            fid = sample_frames(64, of.shape[0], self.sampling)
            of = of[fid]
        
        # # way2. fix length 
        # fid = list(range(of.size(0)))
        # vlen = len(fid)
        # while vlen < 32: # duplicate frames
        #     fid = [f for ind in fid for f in (ind, ind)]
        #     vlen = len(fid)
        # idx = sample_frames(32, vlen, self.sampling)
        # fid = [fid[x] for x in idx]
        # of = of[fid]

        return of

    def get_frames(self, video_name, video_path, keyframe=False, start_ratio=0.0, end_ratio=1.0):
        v = os.path.join(video_path, video_name+'.mp4')
        frames = read_videos_av(v, 32, self.sampling, 1., keyframe=keyframe, start_ratio=start_ratio, end_ratio=end_ratio) # C T H W
        # frames = read_videos(v, 32, self.sampling, 1., keyframe=keyframe, start_ratio=start_ratio, end_ratio=end_ratio)
        frames = self.video_transform(frames)
        frames = frames.permute(1,0,2,3) # T C H W
        return frames