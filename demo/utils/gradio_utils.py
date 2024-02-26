import torch
from transformers import StoppingCriteria

from .prompt import conv_templates, SeparatorStyle
from .model_utils import get_model_name_from_path, KeywordsStoppingCriteria, tokenizer_X_token, disable_torch_init
from .builder_utils import load_model, get_frames, show_frames, flow_to_image

X_TOKEN_INDEX = {'IMAGE': -200, 'VIDEO': -201, 'AUDIO': -202, 'THERMAL': -203, 'DEPTH': -204}
X_INDEX_TOKEN = {v: k for k, v in X_TOKEN_INDEX.items()}

title_markdown = ("""
<div style="display: flex; justify-content: center; align-items: center; text-align: center;">
  <a href="https://github.com/patrick-tssn/LSTP-Chat" style="margin-right: 20px; text-decoration: none; display: flex; align-items: center;">
    <img src="./file=demo/assets/profile.png" alt="LSTP-Chat" style="max-width: 120px; height: auto;">
  </a>
  <div>
    <h1 >LSTP-Chat: Language-guided Spatial-Temporal Prompt Learning for Video Chat</h1>
    <h5 style="margin: 0;">💁 Enjoy.</h5>
  </div>
</div>


<div align="center">
    <div style="display:flex; gap: 0.25rem;" align="center">
        <a href='https://github.com/patrick-tssn/LSTP-Chat'><img src='https://img.shields.io/badge/Github-Code-blue'></a>
        <a href='https://github.com/patrick-tssn/LSTP-Chat/stargazers'><img src='https://img.shields.io/github/stars/patrick-tssn/LSTP-Chat.svg?style=social'></a>
    </div>
</div>
""")

block_css = """
#buttons button {
    min-width: min(120px,100%);
}
"""


tos_markdown = ("""
### Terms of use
research only
""")


learn_more_markdown = ("""
### License
MIT-License
""")


class Chat:
    def __init__(self, model_path, conv_mode, model_base=None, sampler_model_base=None, load_8bit=False, load_4bit=False, device='cuda'):
        disable_torch_init()
        
        ## build model
        model, processor, sampler_processor = load_model(model_path, model_base, sampler_model_base, device=device)
        self.model = model
        self.processor = processor
        self.sampler_processor = sampler_processor
        self.conv_mode = conv_mode
        self.device = device
        # print(self.model)

    def get_prompt(self, qs, state):
        state.append_message(state.roles[0], qs)
        state.append_message(state.roles[1], None)
        return state

    @torch.inference_mode()
    def generate(self, frames, prompt: str, first_run: bool, state, nframe, min_len, max_len, sampling, top_p, beam_size, len_penalty, repetition_penalty):
        model, processor, sampler_processor = self.model, self.processor, self.sampler_processor
        # if model.model.config.pad_token_id == -1: # vicuna1
        #     model.model.config.pad_token_id = processor.tokenizer.pad_token_id

        frames = frames[0][-1]
        # print("================before====================")
        # print("BEFORE PROMPT: \n", prompt)
        # print("================before====================")
        state = self.get_prompt(prompt, state)
        prompt = state.get_prompt()
        
        print('\n\n\n')
        print("================after====================")
        print("AFTER PROMPT:\n", prompt)
        print("================after====================")

        text_encoding = processor(
            text=prompt,
            padding=True,
            return_tensors="pt",
        ).to(self.device)
        sampler_text_encoding = sampler_processor(
            text=prompt,
            padding=True,
            return_tensors="pt"
        ).to(self.device)
        # text_encoding = processor(prompt).to(self.device)
        # sampler_text_encoding = sampler_processor(prompt).to(self.device)        
        
        # temperature = 0.2
        # max_new_tokens = 256
        # nframe = 4
        # repetition_penalty = 1.5
        # do_sample = True

        # nframe, min_len, max_len, sampling, top_p, beam_size, len_penalty, repetition_penalty = hyperparameters

        stop_str = conv_templates[self.conv_mode].copy().sep if conv_templates[self.conv_mode].copy().sep_style != SeparatorStyle.TWO else \
        conv_templates[self.conv_mode].copy().sep2
        keywords = [stop_str]
        with torch.inference_mode():
            output_ids, sampled_indices, frame_feats = model.generate(
                frames,
                nframe,
                text_encoding,
                sampler_text_encoding,
                # do_sample=do_sample,
                # temperature=temperature,
                # max_new_tokens=max_new_tokens,
                sampling=sampling,
                min_len=min_len,
                max_len=max_len,
                num_beams=beam_size,
                top_p=top_p,
                length_penalty=len_penalty,
                repetition_penalty=repetition_penalty,
            )

        outputs = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
        outputs = outputs.strip()
        outputs = outputs.split("</s>")[0]
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()

        print("====================================")
        print('RESPONSE:\n', outputs)
        print('INDICES:\n', sampled_indices.tolist())
        print("====================================")
        return outputs, sampled_indices, frame_feats, state

