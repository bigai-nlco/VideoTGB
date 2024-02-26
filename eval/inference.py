import os
import json
import argparse
import math
from tqdm import tqdm

import torch
from transformers import TextStreamer

import sys
print(sys.path)

from src.data.components.conversation import conv_templates, SeparatorStyle
from src.data.components.constants import DEFAULT_X_START_TOKEN, DEFAULT_X_TOKEN, DEFAULT_X_END_TOKEN, X_TOKEN_INDEX

from .utils.builder_utils import load_pretrained_model, get_frames, KeywordsStoppingCriteria




def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()

    # Define the command-line arguments
    parser.add_argument('--model_path', help='', required=True)
    parser.add_argument('--cache_dir', help='', required=True)
    parser.add_argument('--video_dir', help='Directory containing video files.', required=True)
    parser.add_argument('--gt_file_question', help='Path to the ground truth file containing question.', required=True)
    parser.add_argument('--gt_file_answers', help='Path to the ground truth file containing answers.', required=True)
    parser.add_argument('--output_dir', help='Directory to save the model results JSON.', required=True)
    parser.add_argument('--output_name', help='Name of the file for storing results JSON.', required=True)
    parser.add_argument("--nframe", type=int, default=4)
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--device", type=str, required=False, default='cuda:0')
    parser.add_argument('--model_base', help='', default=None, type=str, required=False)
    parser.add_argument('--sampler_base', help='', default=None, type=str, required=False)
    parser.add_argument("--model_max_length", type=int, required=False, default=2048)
    parser.add_argument("--lora", type=int, default=0)

    return parser.parse_args()

def get_model_name_from_path(model_path):
    model_path = model_path.strip("/")
    model_paths = model_path.split("/")
    if model_paths[-1].startswith('checkpoint-'):
        return model_paths[-2] + "_" + model_paths[-1]
    else:
        return model_paths[-1]

def get_model_output(model, processor, sampler_processor, video_path, question, args):
    
    frames, flow_frames = get_frames(video_path, fps=2)
    # frames = frames.unsqueeze(0)
    flow_frames = flow_frames.unsqueeze(0)

    frames = frames.to(args.device)
    flow_frames = flow_frames.to(args.device)

    # prompt = "question: " + question + "short answer: "
    # prompt = "Question: " + question + "\nAnswer the question using a single word or phrase."
    prompt = "USER: <video>\n" + question + " ASSISTANT: "
    text_encoding = processor(
        text=prompt,
        padding="longest",
        truncation=True,
        max_length=128,
        return_tensors="pt",
    ).to(args.device)
    sampler_text_encoding = sampler_processor(
        text=question,
        padding="longest",
        truncation=True,
        max_length=128,
        return_tensors="pt",
    ).to(args.device)

    if "vicuna" in processor.tokenizer.name_or_path:
        stopping_criteria = [KeywordsStoppingCriteria(['</s>'], processor.tokenizer, text_encoding.input_ids)]
        # stopping_criteria = None
    else:
        stopping_criteria = None

    with torch.inference_mode():
        output_ids, sampled_indices = model.generate(
            frames,
            flow_frames,
            args.nframe,
            text_encoding,
            sampler_text_encoding,
            do_sample=True,
            temperature=0.2,
            max_new_tokens=128,
            use_cache=False,
            stopping_criteria=stopping_criteria,
        )
    if 'vicuna' in processor.tokenizer.name_or_path:
        outputs = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
        if outputs.endswith('</s>'):
            outputs = outputs[:-len('</s>')]
    else:
        outputs = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
    outputs = outputs.strip()
    # print("question: ", question)
    # print("prediciton: ", outputs)
    return outputs

def run_inference(args):
    """
    Run inference on ActivityNet QA DataSet using the Video-ChatGPT model.

    Args:
        args: Command-line arguments.
    """
    # Initialize the model
    # model_name = get_model_name_from_path(args.model_path)
    # model, processor, sampler_processor = load_pretrained_model(args.model_path, args.model_base, args.sampler_base, args.device)
    model, processor, sampler_processor = load_pretrained_model(args.model_path, args.model_base, args.sampler_base, args.device, args.lora)
    model = model.to(args.device)

    # # preprocess data
    # Load both ground truth file containing questions and answers
    # with open(args.gt_file_question) as file:
    #     gt_questions = json.load(file)
    # with open(args.gt_file_answers) as file:
    #     gt_answers = json.load(file)

    gt_questions = json.load(open(args.gt_file_question, "r"))
    gt_questions = get_chunk(gt_questions, args.num_chunks, args.chunk_idx)
    gt_answers = json.load(open(args.gt_file_answers, "r"))
    gt_answers = get_chunk(gt_answers, args.num_chunks, args.chunk_idx)

    answers_file = os.path.join(args.output_dir, f"{args.output_name}.json")
    os.makedirs(args.output_dir, exist_ok=True)
    ans_file = open(answers_file, "w")
    
    # Create the output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    output_list = []  # List to store the output results


    video_formats = ['.mp4', '.avi', '.mov', '.mkv']

    # Iterate over each sample in the ground truth file
    index = 0
    for sample in tqdm(gt_questions):
        video_name = sample['video_name']
        question = sample['question']
        id = sample['question_id']
        answer = gt_answers[index]['answer']
        index += 1

        sample_set = {'id': id, 'question': question, 'answer': answer}

        # Load the video file
        for fmt in video_formats:  # Added this line
            if "Activitynet" in args.video_dir:
                temp_path = os.path.join(args.video_dir, f"v_{video_name}{fmt}")
            else:
                temp_path = os.path.join(args.video_dir, f"{video_name}{fmt}")
            if os.path.exists(temp_path):
                video_path = temp_path
                # try:
                # Run inference on the video and add the output to the list
                output = get_model_output(model, processor, sampler_processor, video_path, question, args)
                sample_set['pred'] = output

                # visualization
                if index % 500 == 0:
                    print("==================CASE====================")
                    print("Question: ", question)
                    print("Answer: ", answer)
                    print("Prediction: ", output)
                    print("==========================================")
                    
                output_list.append(sample_set)
                # except Exception as e:
                #     print(f"Error processing video file '{video_name}': {e}")
                ans_file.write(json.dumps(sample_set) + "\n")
                break

    ans_file.close()
    # Save the output list to a JSON file
    # with open(os.path.join(args.output_dir, f"{args.output_name}.json"), 'w') as file:
    #     json.dump(output_list, file)


if __name__ == "__main__":
    args = parse_args()
    run_inference(args)

