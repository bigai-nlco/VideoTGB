import os
import json
import random
from PIL import Image, ImageFile
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, AutoProcessor, InstructBlipProcessor, BlipImageProcessor

from .util import read_videos, read_videos_cv2, read_videos_av, sample_frames, flow_to_image
from src.data.components import conversation as conversation_lib
from src.data.components.constants import IGNORE_INDEX, X_TOKEN_INDEX, DEFAULT_X_TOKEN, DEFAULT_X_START_TOKEN, DEFAULT_X_END_TOKEN

# ImageFile.LOAD_TRUNCATED_IMAGES = True

class IVINSTRUCT(Dataset):
    
    def __init__(self,
            text_dir: str,
            image_dir: str,
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
        
        self.image_dir = image_dir
        self.video_dir = video_dir
        self.of_dir = of_dir
        self.text_dir = text_dir
        self.max_sampler_txt_len = 128
        self.max_txt_len = 512
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
        
        try:
            data_dct = self.data[index]

            # unify special token : <image>\nxxxx OR <video>\nxxx
            conversations = data_dct["conversations"]
            for sentence in conversations:
                for DEFAULT_TOKEN in DEFAULT_X_TOKEN.values():
                    if DEFAULT_TOKEN in sentence["value"]:
                        sentence["value"] = sentence["value"].replace(DEFAULT_TOKEN,  '').strip()
                        sentence["value"] = DEFAULT_TOKEN + '\n' + sentence["value"]
                        sentence["value"] = sentence["value"].strip()
            
            # apply prompt templates: vicuna v1
            is_vicuna = 0
            if "vicuna" in  self.processor.tokenizer.name_or_path:
                is_vicuna = 1
            question = ""
            answer = ""
            conv = conversation_lib.default_conversation.copy()
            seps = [conv.sep, conv.sep2]
            roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
            if roles[conversations[0]["from"]] != conv.roles[0]:
                conversations = conversations[1:] # skip that the first one is not from human
            for i, sentence in enumerate(conversations):
                role = roles[sentence["from"]]
                assert role == conv.roles[i%2]
                if i == len(conversations)-1:
                    question += role + ": "
                    answer += sentence["value"]
                    if is_vicuna: answer += seps[-1]
                else:
                    question += role + ": " + sentence["value"] + " "
                    if sentence["value"] : question += seps[i%2]

            if 'image' in data_dct:
                
                idx = data_dct["id"]
                image_file = self.data[index]["image"]
                image = self.get_image(image_file, self.image_dir)
                frames = self.image_transform(image).unsqueeze(0)
                width = 1
                start, end = 0, 0


            elif 'video' in data_dct:   
                
                # vid = data_dct["video_id"]
                video_file = self.data[index]["video"]
                # process video and of
                frames = self.get_frames(video_file, self.video_dir) # TCHW
                
                # sample
                video_length = frames.shape[0]
                idx = data_dct["id"]
                # start = int(self.pseudo_label[idx][0] / 31 * (video_length-1))
                # end = int(self.pseudo_label[idx][1] / 31 * (video_length-1))
                start = int(self.pseudo_label[idx][0] * (video_length-1))
                end = int(self.pseudo_label[idx][1] * (video_length-1))
                frames = frames[start: end+1]
                video_length = frames.shape[0]
                fid = sample_frames(self.nframe, video_length, self.sampling)
                frames = frames[fid]
                width = frames.shape[0]

            return {'frames':frames, 'question':question, 'answer':answer, "idx": idx, "start": start, "end": end, "width": width}

        except Exception as e:
            print(f"Error: {e}")
            return self.__getitem__(random.randint(0, self.__len__()-1))

    def collate(self, batch):
        
        # frames = [x['frames'] for x in batch]
        frames = torch.cat([data['frames'] for data in batch]) # B*T, 3, 224, 224
        # calculate frame index
        widths = [data['width'] for data in batch]
        
        answers = [x['answer'] for x in batch]
        questions = [x['question'] for x in batch]
        
        # preprocess text
        sampler_question_encoding = self.sampler_processor(
            text=questions,
            padding='longest',
            truncation=True,
            max_length=self.max_sampler_txt_len,
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
        
        if "instructblip" in self.processor.tokenizer.name_or_path:
            return {"frames": frames, 
                    "sampler_question": sampler_question_encoding["input_ids"],
                    "sampler_question_attention_mask": sampler_question_encoding["attention_mask"],
                    "question": question_encoding["input_ids"],
                    "question_attention_mask": question_encoding["attention_mask"],
                    "qformer_text":question_encoding["qformer_input_ids"],
                    "qformer_text_attention_mask": question_encoding["qformer_attention_mask"],
                    "answer": answer_encoding["input_ids"],
                    "answer_attention_mask": answer_encoding["attention_mask"],
                    "text_answer": answers,
                    "nframe": self.nframe,
                    "widths": widths,
                    }
        elif "blip2" in self.processor.tokenizer.name_or_path:
            return {"frames": frames, 
                "sampler_question": sampler_question_encoding["input_ids"],
                "sampler_question_attention_mask": sampler_question_encoding["attention_mask"],
                "question": question_encoding["input_ids"],
                "question_attention_mask": question_encoding["attention_mask"],
                # "qformer_text":question_encoding["qformer_input_ids"],
                # "qformer_text_attention_mask": question_encoding["qformer_attention_mask"],
                "answer": answer_encoding["input_ids"],
                "answer_attention_mask": answer_encoding["attention_mask"],
                "text_answer": answers,
                "nframe": self.nframe,
                "widths": widths,
                }
            
    
    def _load_data(self):
        
        data_path = os.path.join(self.text_dir, self.split + '.json')
        with open(data_path) as jp:
            # data_dct = json.load(jp)
            data_lst = json.load(jp)
        # data_lst = []
        # for idx, dct in data_dct.items():
        #     dct['idx'] = idx
        #     data_lst.append(dct)
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
        v = os.path.join(video_path, video_name)
        # frames = read_videos_av(v, 32, self.sampling, 1., keyframe=keyframe, start_ratio=start_ratio, end_ratio=end_ratio) # C T H W
        frames = read_videos(v, 32, self.sampling, 1., keyframe=keyframe, start_ratio=start_ratio, end_ratio=end_ratio)
        frames = self.video_transform(frames)
        frames = frames.permute(1,0,2,3) # T C H W
        return frames
    
    def get_image(self, image_name, image_path):
        image_path = os.path.join(image_path, image_name)
        image = Image.open(image_path).convert('RGB')
        return image