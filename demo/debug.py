import time
import numpy as np
from .utils.gradio_utils import flow_to_image, get_frames

flow_path = '/scratch2/nlp/patrick/data/videoinstruction/flow/v_IeTMYNbQSp0_raft.npy'
video_path = '/home/wangyuxuan1/codes/LSTP-Chat/demo/examples/v_GPl7nFwqSgk.mp4'
start = time.time()
flows = np.load(flow_path)
load = time.time()
print('load time: ', load-start)
start = time.time()
for flow in flows:
    flow = flow.transpose(1,2,0)
    flow = flow_to_image(flow)
change = time.time()
print('change time: ', change-start)
start = time.time()
tensor = get_frames(video_path)
sample = time.time()
print('sample time: ', sample-start)

"""RESULTS
load time:  0.005164623260498047
change time:  0.10443615913391113
sample time:  19.30803680419922
"""