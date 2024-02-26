"""Borrow from https://github.com/m-bain/frozen-in-time/blob/main/base/base_dataset.py
"""

import glob
import os.path as op
import random
import av
import cv2
import decord
from decord import cpu
decord.bridge.set_bridge("torch")
import json

import numpy as np
import pandas as pd
import ffmpeg
import torch


def sample_frames(num_frames, video_len, sample='rand', fix_start=-1):
    if num_frames >= video_len:
        return range(video_len)

    intv = np.linspace(start=0, stop=video_len, num=num_frames+1).astype(int)
    if sample == 'rand':
        frame_ids = [random.randrange(intv[i], intv[i+1]) for i in range(len(intv)-1)]
    elif fix_start >= 0:
        fix_start = int(fix_start)
        frame_ids = [intv[i]+fix_start for i in range(len(intv)-1)]
    elif sample == 'uniform':
        frame_ids = [(intv[i]+intv[i+1]-1) // 2 for i in range(len(intv)-1)]
    else:
        raise NotImplementedError
    return frame_ids


def read_frames(video_path, num_frames=-1, sample='rand', trim=1., fix_start=-1, keyframe=False, start_ratio=0.0, end_ratio=1.0, format='png', cand_ratio=None):
    if format == 'png':
        frames = glob.glob(op.join(video_path, '*.png'))
    elif format == 'jpg':
        frames = glob.glob(op.join(video_path, '*.jpg'))
    
    if not len(frames):
        raise FileNotFoundError("No such videos:", video_path)
    # frames.sort(key=lambda n: int(op.basename(n)[:-4]))
    frames.sort(key=lambda n: int(op.basename(n)[6:-4]))

    if trim < 1.:
        remain = (1. - trim) / 2
        start, end = int(len(frames) * remain), int(len(frames) * (1 - remain))
        frames = frames[start:end]
    if keyframe:
        if cand_ratio is None:
            start, end = int(len(frames)*start_ratio), int(len(frames)*end_ratio)+1
            frames = frames[start:end]
        else:
            cand_frames = []
            for cand in cand_ratio:
                start, end = int(len(frames)*cand[0]), int(len(frames)*cand[1])+1
                cand_frames += frames[start: end]
            cand_frames = list(set(cand_frames))
            cand_frames.sort(key=lambda n:int(op.basename(n)[:-4]))
            print(len(cand_frames), len(frames))
            frames = cand_frames
                
    if num_frames > 0:
        while len(frames) < num_frames: # duplicate frames
            frames = [f for frame in frames for f in (frame, frame)]
        frame_ids = sample_frames(num_frames, len(frames), sample, fix_start)
        return [frames[i] for i in frame_ids]
    return frames

def read_videos(video_path, num_frames=-1, sample='rand', trim=1., fix_start=-1, keyframe=False, start_ratio=0.0, end_ratio=1.0):
    # video_reader = decord.VideoReader(video_path, height=224, width=224)
    # video_reader = decord.VideoReader(video_path, num_threads=0)
    video_reader = decord.VideoReader(video_path, ctx=cpu(0))
    vlen = len(video_reader)

    ori_indices = list(range(vlen))
    
    if trim < 1.:
        remain = (1. - trim) / 2
        start, end = int(vlen * remain), int(vlen * (1 - remain))
        indices = ori_indices[start:end]
    if keyframe:
        start, end = int(vlen*start_ratio), int(vlen*end_ratio)+1
        indices = ori_indices[start:end]

    if num_frames > 0:
        while vlen < num_frames: # duplicate frames
            ori_indices = [f for ind in ori_indices for f in (ind, ind)]
            vlen = len(ori_indices)
        frame_ids = sample_frames(num_frames, vlen, sample, fix_start)
        indices = [ori_indices[ii] for ii in frame_ids]
    frames = video_reader.get_batch(indices).permute(3,0,1,2).float() # T H W C -> C T H W
    return frames

def read_videos_cv2(video_path, num_frames=-1, sample='rand', trim=1., fix_start=-1, keyframe=False, start_ratio=0.0, end_ratio=1.0):
    video = cv2.VideoCapture(video_path)
    success, image = video.read()
    success = True
    frames = []
    while success:
        frames.append(image)
        success, image = video.read()
    vlen = len(frames)
    ori_indices = list(range(vlen))
    
    if trim < 1.:
        remain = (1. - trim) / 2
        start, end = int(vlen * remain), int(vlen * (1 - remain))
        indices = ori_indices[start:end]
    if keyframe:
        start, end = int(vlen*start_ratio), int(vlen*end_ratio)+1
        indices = ori_indices[start:end]

    if num_frames > 0:
        while vlen < num_frames: # duplicate frames
            ori_indices = [f for ind in ori_indices for f in (ind, ind)]
            vlen = len(ori_indices)
        frame_ids = sample_frames(num_frames, vlen, sample, fix_start)
        indices = [ori_indices[ii] for ii in frame_ids]
    frames = torch.from_numpy(np.stack([frames[x] for x in indices], axis=0)).permute(3,0,1,2).float() # T H W C -> C T H W
    video.release()
    return frames 

def read_videos_av(video_path, num_frames=-1, sample='rand', trim=1., fix_start=-1, keyframe=False, start_ratio=0.0, end_ratio=1.0, fps=None):
    video = av.open(video_path)
    frames = []

    if fps is not None:
        avg_fps = int(video.streams.video[0].averate_rate)
        if fps <= avg_fps:
            step = avg_fps
            for idx, frame in enumerate(video.decode(video=0)):
                if idx % step == 0:
                    frame = frame.to_ndarray(format='rgb24')
                    frames.append(frame)

    for frame in video.decode(video=0):
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

    if num_frames > 0:
        while vlen < num_frames: # duplicate frames
            ori_indices = [f for ind in ori_indices for f in (ind, ind)]
            vlen = len(ori_indices)
        frame_ids = sample_frames(num_frames, vlen, sample, fix_start)
        indices = [ori_indices[ii] for ii in frame_ids]
    frames = torch.from_numpy(np.stack([frames[x] for x in indices], axis=0)).permute(3,0,1,2).float() # T H W C -> C T H W
    return frames

def load_file(filename):
    """
    load obj from filename
    :param filename:
    :return:
    """
    cont = None
    if not op.exists(filename):
        print('{} not exist'.format(filename))
        return cont
    if op.splitext(filename)[-1] == '.csv':
        return pd.read_csv(filename, delimiter=',')
    with open(filename, 'r') as fp:
        if op.splitext(filename)[1] == '.txt':
            cont = fp.readlines()
            cont = [c.rstrip('\n') for c in cont]
        elif op.splitext(filename)[1] == '.json':
            cont = json.load(fp)
    return cont

# convert flow to rgb
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