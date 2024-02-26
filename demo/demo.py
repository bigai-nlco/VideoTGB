import shutil
import subprocess
import gradio as gr
from fastapi import FastAPI
import os
from PIL import Image
import tempfile
import cv2
# from decord import VideoReader, cpu

import torch
from transformers import TextStreamer

from .utils.prompt import conv_templates, Conversation
from .utils.gradio_utils import Chat, get_frames, show_frames, flow_to_image, tos_markdown, learn_more_markdown, title_markdown, block_css

DEFAULT_X_TOKEN = {'IMAGE': "<image>", 'VIDEO': "<video>", 'AUDIO': "<audio>", 'THERMAL': "<thermal>", 'DEPTH': "<depth>"}

def save_image_to_local(image):
    filename = os.path.join('temp', next(tempfile._get_candidate_names()) + '.jpg')
    image = Image.open(image)
    image.save(filename)
    # print(filename)
    return filename


def save_video_to_local(video_path):
    filename = os.path.join('temp', next(tempfile._get_candidate_names()) + '.mp4')
    shutil.copyfile(video_path, filename)
    return filename

def copy_image_to_local(image):
    filename = os.path.join('temp', next(tempfile._get_candidate_names()) + '.jpg')
    image.save(filename)
    return filename

def copy_flow_to_local(flow):
    filename = os.path.join('temp', next(tempfile._get_candidate_names()) + '.png')
    cv2.imwrite(filename, flow[:, :, [2, 1, 0]])
    return filename

def generate(video, textbox_in, first_run, state, state_, images_tensor, nframe, min_len, max_len, sampling, top_p, beam_size, len_penalty, repetition_penalty):
    flag = 1
    if not textbox_in:
        if len(state_.messages) > 0:
            textbox_in = state_.messages[-1][1]
            state_.messages.pop(-1)
            flag = 0
        else:
            return "Please enter instruction"

    if type(state) is not Conversation:
        state = conv_templates[conv_mode].copy()
        state_ = conv_templates[conv_mode].copy()
        images_tensor = [[], []]

    first_run = False if len(state.messages) > 0 else True

    # text_en_in = textbox_in.replace("picture", "image")
    text_en_in = "<video>\n" + textbox_in

    if first_run:
        tensor, frames = get_frames(video)
        tensor = tensor.to(handler.model.device, dtype=dtype)
        tensor = tensor.unsqueeze(0)
        images_tensor[0] += tensor
        # frames = show_frames(video)
        images_tensor[1].append(frames)

    # text_en_in = DEFAULT_X_TOKEN['VIDEO'] + '\n' + text_en_in
    
    text_en_out, sampled_indices, of_feats, state_ = handler.generate(images_tensor, text_en_in, first_run=first_run, state=state_, nframe=nframe, min_len=min_len, max_len=max_len, sampling=sampling, top_p=top_p, beam_size=beam_size, len_penalty=len_penalty, repetition_penalty=repetition_penalty)
    state_.messages[-1] = (state_.roles[1], text_en_out)

    text_en_out = text_en_out.split('#')[0]
    textbox_out = text_en_out

    if first_run:
        show_images = ""
    #     filename = save_video_to_local(video)
    #     show_images += f'<video controls playsinline width="500" style="display: inline-block;"  src="./file={filename}"></video>'

    # show sampled images
    show_sampled_images = ""
    sampled_frames = [images_tensor[1][0][idx] for idx in sampled_indices]
    # images_tensor[1].append(sampled_frames) # for further process
    for frame in sampled_frames:
        frame = copy_image_to_local(frame)
        show_sampled_images += f'<img src="./file={frame}" style="display: inline-block;width: 100px;max-height: 200px;">'
    show_sampled_images = ">>>> Selected Frames: \n" + show_sampled_images + "\n" + ">>>> Corresponding Flows: \n"

    # show optical flows
    for flow in of_feats:
        flow = flow.permute(1,2,0).cpu().numpy() # hw2
        flow = flow_to_image(flow)
        flow = copy_flow_to_local(flow)
        show_sampled_images += f'<img src="./file={flow}" style="display: inline-block;width: 100px;max-height: 200px;">'
    

    if flag:
        if first_run:
            state.append_message(state.roles[0], textbox_in + "\n" + show_images)
        else:
            state.append_message(state.roles[0], textbox_in)
    state.append_message(state.roles[1], textbox_out + '\n' + show_sampled_images)

    return (state, state_, state.to_gradio_chatbot(), False, gr.update(value=None, interactive=True), images_tensor, gr.update(value=video if os.path.exists(video) else None, interactive=True))

def regenerate(state, state_):
    state.messages.pop(-1)
    state_.messages.pop(-1)
    if len(state.messages) > 0:
        return state, state_, state.to_gradio_chatbot(), False
    return (state, state_, state.to_gradio_chatbot(), True)


def clear_history(state, state_):
    state = conv_templates[conv_mode].copy()
    state_ = conv_templates[conv_mode].copy()
    return (gr.update(value=None, interactive=True),\
        gr.update(value=None, interactive=True),\
        True, state, state_, state.to_gradio_chatbot(), [[], []])




conv_mode = "lstp"
model_path = "ckpts/LSTP-Chat/LSTP-7B.ckpt"
model_base = 'ckpts/instructblip-vicuna-7b'

# model_path = 'ckpts/LSTP-Chat/xxx.ckpt'
# model_base = 'ckpts/blip2-flan-t5-xl'

sampler_model_base = 'ckpts/bert-base-uncased'
device = 'cuda:7'
load_8bit = False
load_4bit = False
dtype = torch.float32
handler = Chat(model_path, conv_mode=conv_mode, model_base=model_base, sampler_model_base=sampler_model_base, load_8bit=load_8bit, load_4bit=load_8bit, device=device)
# handler.model.to(dtype=dtype)
if not os.path.exists("temp"):
    os.makedirs("temp")

# app = FastAPI()
    
# hyperparameters


# hyperparameters = [nframe, min_len, max_len, sampling, top_p, beam_size, len_penalty, repetition_penalty]

textbox = gr.Textbox(
        show_label=False, placeholder="Enter text and press ENTER", container=False
    )
with gr.Blocks(title='LSTP-Chat', theme=gr.themes.Default(), css=block_css) as demo:
    gr.Markdown(title_markdown)
    state = gr.State()
    state_ = gr.State()
    first_run = gr.State()
    images_tensor = gr.State()

    with gr.Row():
        with gr.Column(scale=3):
            video = gr.Video(label="Input Video")

            cur_dir = os.path.dirname(os.path.abspath(__file__))
            print(cur_dir)
            gr.Examples(
                examples=[
                    [
                        f"{cur_dir}/examples/sample_demo_1.mp4",
                        "Why is this video funny?"
                    ],
                    [
                        f"{cur_dir}/examples/sample_demo_3.mp4",
                        "Can you identify any safety hazards in this video?",
                    ],
                    [
                        f"{cur_dir}/examples/sample_demo_8.mp4",
                        "Describe the video."
                    ],
                    [
                        f"{cur_dir}/examples/sample_demo_9.mp4",
                        "Describe the activity in the video."
                    ]
                ],
                inputs=[video, textbox],
            )

        with gr.Column(scale=7):
            chatbot = gr.Chatbot(label="LSTP-Chat", bubble_full_width=True).style(height=700)
            with gr.Row():
                with gr.Column(scale=8):
                    textbox.render()
                with gr.Column(scale=1, min_width=50):
                    submit_btn = gr.Button(
                        value="Send", variant="primary", interactive=True
                    )
            with gr.Row(elem_id="buttons") as button_row:
                upvote_btn = gr.Button(value="🔼  Upvote", interactive=True) # unimplemented
                downvote_btn = gr.Button(value="🔽  Downvote", interactive=True) # unimplemented
                # stop_btn = gr.Button(value="⏹️  Stop Generation", interactive=False)
                regenerate_btn = gr.Button(value="↩️  Regenerate", interactive=True)
                clear_btn = gr.Button(value="🔄  Reset", interactive=True)

        with gr.Accordion("Parameters", open=False) as parameter_row:
            nframe = gr.Slider(
                minimum=1,
                maximum=8,
                value=4,
                step=1,
                interactive=True,
                label="Number of Frames",
            )

            min_len = gr.Slider(
                minimum=1,
                maximum=50,
                value=1,
                step=1,
                interactive=True,
                label="Min Length",
            )

            max_len = gr.Slider(
                minimum=10,
                maximum=500,
                value=128,
                step=5,
                interactive=True,
                label="Max Length",
            )

            sampling = gr.Radio(
                choices=["Beam search", "Nucleus sampling"],
                value="Beam search",
                label="Text Decoding Method",
                interactive=True,
            )

            top_p = gr.Slider(
                minimum=0.5,
                maximum=1.0,
                value=0.9,
                step=0.1,
                interactive=True,
                label="Top p",
            )

            beam_size = gr.Slider(
                minimum=1,
                maximum=10,
                value=5,
                step=1,
                interactive=True,
                label="Beam Size",
            )

            len_penalty = gr.Slider(
                minimum=-1,
                maximum=2,
                value=1,
                step=0.2,
                interactive=True,
                label="Length Penalty",
            )

            repetition_penalty = gr.Slider(
                minimum=-1,
                maximum=3,
                value=1,
                step=0.2,
                interactive=True,
                label="Repetition Penalty",
            )


    gr.Markdown(tos_markdown)
    gr.Markdown(learn_more_markdown)

    submit_btn.click(generate, [video, textbox, first_run, state, state_, images_tensor, nframe, min_len, max_len, sampling, top_p, beam_size, len_penalty, repetition_penalty],
                     [state, state_, chatbot, first_run, textbox, images_tensor, video])

    regenerate_btn.click(regenerate, [state, state_], [state, state_, chatbot, first_run]).then(
        generate, [video, textbox, first_run, state, state_, images_tensor, nframe, min_len, max_len, sampling, top_p, beam_size, len_penalty, repetition_penalty], [state, state_, chatbot, first_run, textbox, images_tensor, video])

    clear_btn.click(clear_history, [state, state_],
                    [video, textbox, first_run, state, state_, chatbot, images_tensor])

# app = gr.mount_gradio_app(app, demo, path="/")
demo.launch(share=False, enable_queue=True, show_api=False)


# uvicorn llava.serve.gradio_web_server:app
