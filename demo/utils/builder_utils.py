import os
import av

import numpy as np
import torch
from torchvision.transforms import Compose
from transformers import AutoProcessor, AutoTokenizer

from src.data.components.util import sample_frames
from src.gadgets.transforms import RandomResizedCropVideo, ToTHWC, ToUint8, ToTensorVideo, NormalizeVideo, ResizeVideo
from .model import LSTP, LSTP_blip2


DEFAULT_X_PATCH_TOKEN = {'IMAGE': "<im_patch>", 'VIDEO': "<vi_patch>", 'AUDIO': "<au_patch>", 'THERMAL': "<th_patch>", 'DEPTH': "<de_patch>"}
# DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_X_START_TOKEN = {'IMAGE': "<im_start>", 'VIDEO': "<vi_start>", 'AUDIO': "<au_start>", 'THERMAL': "<th_start>", 'DEPTH': "<de_start>"}
# DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_X_END_TOKEN = {'IMAGE': "<im_end>", 'VIDEO': "<vi_end>", 'AUDIO': "<au_end>", 'THERMAL': "<th_end>", 'DEPTH': "<de_end>"}
# DEFAULT_IM_END_TOKEN = "<im_end>"

def read_videos(video_path, num_frames=-1, sample='rand', trim=1., fix_start=-1, keyframe=False, start_ratio=0.0, end_ratio=1.0):
    video = av.open(video_path)
    frames = []
    for frame in video.decode(video=0):
        frame = frame.to_image()
        frames.append(frame)
    vlen = len(frames)
    ori_indices = list(range(vlen))
    indices = list(range(vlen))
    
    if trim < 1.:
        remain = (1. - trim) / 2
        start, end = int(vlen * remain), int(vlen * (1 - remain))
        indices = ori_indices[start:end]
    if keyframe:
        start, end = int(vlen*start_ratio), int(vlen*end_ratio)+1
        indices = ori_indices[start:end]

    if num_frames > 0 and vlen > num_frames:
        while vlen < num_frames: # duplicate frames
            ori_indices = [f for ind in ori_indices for f in (ind, ind)]
            vlen = len(ori_indices)
        frame_ids = sample_frames(num_frames, vlen, sample, fix_start)
        indices = [ori_indices[ii] for ii in frame_ids]
    return [frames[x] for x in indices]

def read_videos_av(video_path, num_frames=-1, sample='rand', trim=1., fix_start=-1, keyframe=False, start_ratio=0.0, end_ratio=1.0, fps=None):
    video = av.open(video_path)
    frames = []
    ori_frames = []

    if fps is not None:
        avg_fps = int(video.streams.video[0].averate_rate)
        if fps <= avg_fps:
            step = avg_fps
            for idx, frame in enumerate(video.decode(video=0)):
                if idx % step == 0:
                    frame = frame.to_ndarray(format='rgb24')
                    frames.append(frame)

    for frame in video.decode(video=0):
        ori_frame = frame.to_image()
        ori_frames.append(ori_frame)
        frame = frame.to_ndarray(format='rgb24')
        frames.append(frame)
    vlen = len(frames)
    ori_indices = list(range(vlen))
    indices = list(range(vlen))
    
    if trim < 1.:
        remain = (1. - trim) / 2
        start, end = int(vlen * remain), int(vlen * (1 - remain))
        indices = ori_indices[start:end]
    if keyframe:
        start, end = int(vlen*start_ratio), int(vlen*end_ratio)+1
        indices = ori_indices[start:end]

    if num_frames > 0 and vlen > num_frames:
        while vlen < num_frames: # duplicate frames
            ori_indices = [f for ind in ori_indices for f in (ind, ind)]
            vlen = len(ori_indices)
        frame_ids = sample_frames(num_frames, vlen, sample, fix_start)
        indices = [ori_indices[ii] for ii in frame_ids]
    ori_frames = [ori_frames[x] for x in indices]
    frames = torch.from_numpy(np.stack([frames[x] for x in indices], axis=0)).permute(3,0,1,2).float() # T H W C -> C T H W
    return frames, ori_frames



def get_frames(video_path, target_size=224, keyframe=False, start_ratio=0.0, end_ratio=1.0):
    video_transform = Compose([
        ResizeVideo(target_size),
        ToUint8(),
        ToTHWC(),
        ToTensorVideo(),
        NormalizeVideo((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),  # C, T, H, W
    ])
    frames, ori_frames = read_videos_av(video_path, 100, "uniform", 1., keyframe=keyframe, start_ratio=start_ratio, end_ratio=end_ratio)
    frames = video_transform(frames)
    frames = frames.permute(1,0,2,3) # T C H W
    return frames, ori_frames

def show_frames(video_path, target_size=224, keyframe=False, start_ratio=0.0, end_ratio=1.0):
    frames = read_videos(video_path, -1, "uniform", 1., keyframe=keyframe, start_ratio=start_ratio, end_ratio=end_ratio)
    return frames

def load_data(text, video, nframe, processor, sampler_processor):
    frames = get_frames(video)
    text_encoding = processor(
        text=text,
        padding=True,
        return_tensors="pt",
    )
    sampler_text_encoding = sampler_processor(
        text=text,
        padding=True,
        return_tensors="pt"
    )
    return frames, text_encoding, sampler_text_encoding

def dp_state_to_normal(state_dict):
    '''Converts a torch.DataParallel checkpoint to regular'''
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module'):
            new_state_dict[k.replace('module.', '')] = v
    return new_state_dict

def load_model(ckpt_path, base_model_path, base_sampler_path, device):
    print("start to load model...")
    processor = AutoProcessor.from_pretrained(base_model_path)
    sampler_processor = AutoTokenizer.from_pretrained(base_sampler_path)

    if "instructblip" in base_model_path:
        model = LSTP(base_model_path, device)
    elif "blip2" in base_model_path:
        model = LSTP_blip2(base_model_path, device)
    state_dict = torch.load(ckpt_path, map_location='cpu')
    # state_dict = dp_state_to_normal(state_dict)
    # msg = model.load_state_dict(state_dict['state_dict'])
    msg = model.load_state_dict(state_dict)
    print(">>> Load checkpoint for LSTP from", ckpt_path)
    miss = set(m.split('.')[0] for m in msg.missing_keys)
    unexp = set(m.split('.')[0] for m in msg.unexpected_keys)
    print("Missing:", miss if len(miss) else "None")
    print("Unexpected:", unexp if len(unexp) else "None")
    model.to(device)

    return model, processor, sampler_processor

# visualize flow
def make_colorwheel():
    """
    Generates a color wheel for optical flow visualization as presented in:
        Baker et al. "A Database and Evaluation Methodology for Optical Flow" (ICCV, 2007)
        URL: http://vision.middlebury.edu/flow/flowEval-iccv07.pdf

    Code follows the original C++ source code of Daniel Scharstein.
    Code follows the the Matlab source code of Deqing Sun.

    Returns:
        np.ndarray: Color wheel
    """

    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3))
    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255*np.arange(0,RY)/RY)
    col = col+RY
    # YG
    colorwheel[col:col+YG, 0] = 255 - np.floor(255*np.arange(0,YG)/YG)
    colorwheel[col:col+YG, 1] = 255
    col = col+YG
    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.floor(255*np.arange(0,GC)/GC)
    col = col+GC
    # CB
    colorwheel[col:col+CB, 1] = 255 - np.floor(255*np.arange(CB)/CB)
    colorwheel[col:col+CB, 2] = 255
    col = col+CB
    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.floor(255*np.arange(0,BM)/BM)
    col = col+BM
    # MR
    colorwheel[col:col+MR, 2] = 255 - np.floor(255*np.arange(MR)/MR)
    colorwheel[col:col+MR, 0] = 255
    return colorwheel


def flow_uv_to_colors(u, v, convert_to_bgr=False):
    """
    Applies the flow color wheel to (possibly clipped) flow components u and v.

    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun

    Args:
        u (np.ndarray): Input horizontal flow of shape [H,W]
        v (np.ndarray): Input vertical flow of shape [H,W]
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    flow_image = np.zeros((u.shape[0], u.shape[1], 3), np.uint8)
    colorwheel = make_colorwheel()  # shape [55x3]
    ncols = colorwheel.shape[0]
    rad = np.sqrt(np.square(u) + np.square(v))
    a = np.arctan2(-v, -u)/np.pi
    fk = (a+1) / 2*(ncols-1)
    k0 = np.floor(fk).astype(np.int32)
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0
    for i in range(colorwheel.shape[1]):
        tmp = colorwheel[:,i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1-f)*col0 + f*col1
        idx = (rad <= 1)
        col[idx]  = 1 - rad[idx] * (1-col[idx])
        col[~idx] = col[~idx] * 0.75   # out of range
        # Note the 2-i => BGR instead of RGB
        ch_idx = 2-i if convert_to_bgr else i
        flow_image[:,:,ch_idx] = np.floor(255 * col)
    return flow_image


def flow_to_image(flow_uv, clip_flow=None, convert_to_bgr=False):
    """
    Expects a two dimensional flow image of shape.

    Args:
        flow_uv (np.ndarray): Flow UV image of shape [H,W,2]
        clip_flow (float, optional): Clip maximum of flow values. Defaults to None.
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    assert flow_uv.ndim == 3, 'input flow must have three dimensions'
    assert flow_uv.shape[2] == 2, 'input flow must have shape [H,W,2]'
    if clip_flow is not None:
        flow_uv = np.clip(flow_uv, 0, clip_flow)
    u = flow_uv[:,:,0]
    v = flow_uv[:,:,1]
    rad = np.sqrt(np.square(u) + np.square(v))
    rad_max = np.max(rad)
    epsilon = 1e-5
    u = u / (rad_max + epsilon)
    v = v / (rad_max + epsilon)
    return flow_uv_to_colors(u, v, convert_to_bgr)