import asyncio
import base64
import torch
import json
import os
import re
import time
import sys
from copy import deepcopy
from io import BytesIO
from multiprocessing import cpu_count
from typing import List, Tuple, Union
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from accelerate import Accelerator, DistributedType
from transformers import AutoTokenizer
from decord import VideoReader, cpu
from loguru import logger as eval_logger
from openai import AsyncOpenAI, OpenAI
from PIL import Image, ImageDraw
from tqdm import tqdm
from qwen_vl_utils import smart_resize
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pdb
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

NUM_SECONDS_TO_SLEEP = 5

#try:
# from verl.workers.rollout.vllm_rollout.function_tools import get_bbox_from_response
import vllm
#except ImportError:
#    vllm = None
IMAGE_CROP_SYSTEM_PROMPT = """You are a helpful assistant.

# Tools
You may call the function tool shown below to assist with the user query.

You are provided with the function signature within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name_for_human": "request_local_region", "name": "request_local_region", "description": "Request a high-resolution local region of the current image and zoom in", "parameters": {"properties": {"bbox_2d": {"type": "array", "items":{"type":"integer"}, "minItems":4, "maxItems":4, "description": "The bounding box of the region to crop, as [x1, y1, x2, y2], where (x1, y1) is the top-left corner of the target region and (x2, y2) is the bottom-right corner of the target region. The bounding box should be in the absolute pixel coordinates of the current image."}}, "required": ["bbox_2d"], "type": "object"}, "args_format": "Format the arguments as a JSON object."}}
</tools>
For each function call, return a json object with the function name and the corresponding argument within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>
"""

IMAGE_CROP_TOOL_CALL_MULTI_TRUN_PROMPT="Please carefully analyze the content returned from the request_local_region tool in combination with the original question and image from the user, continue your reasoning process inside <think> and </think> and then write your final answer inside <answer> and </answer>."
ERROR_INFO_MULTI_TURN_PROMPT="Please analyze the error information obtained from the function tool and adjust your response. Countinue your reasoning process inside <think> and </think>."

def resize_image(image: Image.Image, save_path=None, ds_factor=2):
    original_width, original_height = image.size
    # Calculate the new dimensions (double the size)
    new_width = original_width * ds_factor
    new_height = original_height * ds_factor
    print(f"[TOOL CALL RESIZE IMAGE]: NEW_IMAGE_SIZE: {(new_width, new_height)}.")
    # Resize the image
    resized_image = image.resize((new_width, new_height), resample=Image.Resampling.LANCZOS)
    if save_path:
        # Save the enlarged image
        resized_image.save(save_path)
    return resized_image

def check_bbox_format(bbox, w, h, MAX_RATIO=200):
    """
    Check whether a bounding box is in valid absolute format within image dimensions.

    Parameters:
        bbox (list): Bounding box in [x0, y0, x1, y1] format.
        w (int or float): Width of the image.
        h (int or float): Height of the image.

    Returns:
        (bool, str): (True, message) if valid; otherwise (False, error message).
    """
    if not isinstance(bbox, list):
        return False, "[WARNING] Bounding box must be a list."

    if len(bbox) != 4:
        return False, f"[WARNING] Bounding box must contain 4 elements: {bbox}"

    if not all(isinstance(coord, (int, float)) for coord in bbox):
        return False, f"[WARNING] All bounding box elements must be numeric: {bbox}"

    x0, y0, x1, y1 = bbox

    if not (0 <= x0 < w and 0 <= y0 < h and 0 < x1 <= w and 0 < y1 <= h):
        return False, f"[WARNING] Bounding box values must be within image bounds [0, 0, {w}, {h}]: {bbox}"

    if x1 <= x0 or y1 <= y0:
        return False, f"[WARNING] Invalid bounding box coordinates: x1 must be > x0 and y1 must be > y0: {bbox}"
    
    height, width = y1 - y0, x1 - x0
    if max(height, width) / min(height, width) > MAX_RATIO:
        return False, f"[WARNING] Bounding box ratio is too large: {max(height, width) / min(height, width)}"

    print(f"[INFO] Valid absolute bounding box: {bbox}")

    return True, "Bounding box format is valid."

TOOL_CALL_PATTERN = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)

@lru_cache(maxsize=1000)
def parse_tool_call_json(tool_call_content: str):
    try:
        tool_call_data = json.loads(tool_call_content)
        if ('name' in tool_call_data and 'arguments' in tool_call_data and 
            'bbox_2d' in tool_call_data['arguments'] and 
            len(tool_call_data['arguments']['bbox_2d']) == 4):
            return tool_call_data["arguments"]["bbox_2d"]
        return None
    except Exception:
        return None

def get_bbox_from_response(response: str, image, ds_factor=2):
    W, H = image.size
    
    tool_call_match = TOOL_CALL_PATTERN.search(response)
    if not tool_call_match:
        print("[WARNING] No tool_call found in outputs_string.")
        return None
    
    tool_call_content = tool_call_match.group(1).strip()
    bbox_content = parse_tool_call_json(tool_call_content)
    
    if bbox_content is None:
        print(f"[WARNING] Invalid bbox in tool_call")
        return None
    
    x0, y0, x1, y1 = bbox_content
    

    scale_factor = ds_factor
    W2, H2 = W * scale_factor, H * scale_factor
    bbox_scaled = [
        max(0, min(int(x0 * scale_factor), W2)),
        max(0, min(int(y0 * scale_factor), H2)),
        max(0, min(int(x1 * scale_factor), W2)),
        max(0, min(int(y1 * scale_factor), H2))
    ]
    
    is_valid, error_msg = check_bbox_format(bbox_scaled, W2, H2)
    if not is_valid:
        print(f"[WARNING] Invalid bbox after clamping: {bbox_scaled} - {error_msg}")
        return None
    
    return bbox_scaled

def save_image_with_bbox_and_question(image: Image.Image, bbox: list, save_path: str, question: str = None, final_answer: str = None):
    """
    保存带有bbox标注、question文本和模型回答的图像。
    
    Args:
        image: PIL Image对象
        bbox: 边界框坐标 [x1, y1, x2, y2]，如果是[0,0,0,0]则不绘制bbox
        save_path: 保存图像的路径
        question: 问题文本，如果提供则会显示在图像上方
        final_answer: 模型回答文本，如果提供则会显示在图像下方
    """
    # 确保保存目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 使用matplotlib创建可视化，参考data_collect_gemini.py的方式
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.imshow(image)
    
    # 只有当bbox不是[0,0,0,0]时才绘制边界框
    x1, y1, x2, y2 = bbox
    if not (x1 == 0 and y1 == 0 and x2 == 0 and y2 == 0):
        width = x2 - x1
        height = y2 - y1
        
        # 创建矩形框
        from matplotlib.patches import Rectangle
        rect = Rectangle((x1, y1), width, height, 
                       linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        
        # 添加bbox坐标标注
        ax.text(x1, y1-10, f'BBOX: [{x1}, {y1}, {x2}, {y2}]', 
               fontsize=10, color='red', weight='bold',
               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # 设置问题作为标题（显示在图像上方，不占用图像区域）
    title_text = ""
    if question:
        title_text += f"Question: {question}"
    if final_answer:
        if title_text:
            title_text += f"\n\nAnswer: {final_answer}"
        else:
            title_text = f"Answer: {final_answer}"
    
    if title_text:
        ax.set_title(title_text, fontsize=12, weight='bold', pad=20, 
                    wrap=True, loc='left')
    
    ax.axis('off')
    plt.tight_layout()
    
    # 保存图像
    plt.savefig(save_path, dpi=150, bbox_inches='tight', 
               facecolor='white', edgecolor='none')
    plt.close()
    
    # 根据是否有bbox调整日志信息
    if x1 == 0 and y1 == 0 and x2 == 0 and y2 == 0:
        print(f"[INFO] 已保存图像（无bbox标注）、问题和回答到: {save_path}")
    else:
        print(f"[INFO] 已保存带bbox标注、问题和回答的图像到: {save_path}")


def crop_image_by_bbox(image: Image.Image, bbox: list):
    """
    使用边界框裁剪图像（优化版本）。
    
    Args:
        image: PIL Image对象
        bbox: 边界框坐标 [x1, y1, x2, y2]
    
    Returns:
        PIL Image: 裁剪并调整大小后的图像
    """
    # 裁剪图像
    cropped_image = image.crop(bbox)
    original_w, original_h = cropped_image.size
    
    new_h, new_w = smart_resize(original_h, original_w, factor=28, min_pixels=4*28*28, max_pixels=3840*3840)
    
    # 优化：只有在尺寸确实需要改变时才进行resize
    if new_w != original_w or new_h != original_h:
        return_image = cropped_image.resize((new_w, new_h), resample=Image.Resampling.LANCZOS)
    else:
        return_image = cropped_image
    
    print(f"[INFO] 裁剪后图像尺寸: {return_image.size}")
    return return_image

def batch_crop_images(images_and_bboxes: List[Tuple[Image.Image, list]]):
    with ThreadPoolExecutor(max_workers=min(4, len(images_and_bboxes))) as executor:
        results = list(executor.map(lambda x: crop_image_by_bbox(x[0], x[1]), images_and_bboxes))
    return results

def prepare_bbox_tool_call_inputs(json_objects: list):
    if not json_objects:
        raise ValueError("Empty tool call: no JSON object found inside <tool_call>.")
    if len(json_objects) != 1:
        raise AssertionError("You should only call function `request_local_region` once per function call.")
    obj = json_objects[0]
    action_type = obj.get('name')
    assert action_type in ["request_local_region"], (
        f"Unknown Tool Type: {action_type}. Available function tools are: `request_local_region`"
    )
    bbox_2d = obj.get('arguments', {}).get('bbox_2d')
    assert bbox_2d is not None, "Missing `bbox_2d` in tool call arguments."
    assert len(bbox_2d) == 4, f"Bounding box must contain 4 elements: {bbox_2d}"    
    assert all(isinstance(coord, (int, float)) for coord in bbox_2d), f"All bounding box elements must be numeric: {bbox_2d}"
    assert bbox_2d[2] > bbox_2d[0] and bbox_2d[3] > bbox_2d[1], (
        f"Invalid bounding box coordinates: x1 must be > x0 and y1 must be > y0: {bbox_2d}"
    )
    return action_type, bbox_2d

@register_model("vllm_bbox")
class VLLM_AdaptVision(lmms):
    def __init__(
        self,
        model_version: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.7,
        batch_size: int = 1,
        timeout: int = 60,
        max_images: int = 32,
        prompt: str = 'tool_call',
        enable_tool_call: bool = False,
        max_generation_round: bool = 2,
        downsample_image: bool = False,
        ds_factor: int = 2,
        save_path: str = None,
        **kwargs,
    ) -> None:
        super().__init__()
        # Manually set a image token for GPT4V so that we can search for it
        # and split the text and image
        # Here we just use the same token as llava for convenient
        self.model_version = model_version
        self.max_images = max_images
        self.enable_tool_call = enable_tool_call
        self.max_generation_round = max_generation_round
        self.downsample_image = downsample_image
        self.ds_factor = ds_factor
        self.save_path = save_path
        accelerator = Accelerator()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_version)
        self.inference_engine = vllm.LLM(
            model=self.model_version,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_num_batched_tokens=32768,
            enable_prefix_caching=False,
            max_model_len=32768,
            limit_mm_per_prompt={"image": max_images}
        )
        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [DistributedType.FSDP, DistributedType.MULTI_GPU, DistributedType.DEEPSPEED], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self.accelerator = accelerator
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes

        self.device = self.accelerator.device
        self.batch_size_per_gpu = int(batch_size)
        self.prompt = prompt
        

        self.bbox_cache = {} 
        self.image_cache = {} 
        self.thread_pool = ThreadPoolExecutor(max_workers=min(4, cpu_count()))
        
        # 确保保存目录存在并输出日志
        if self.save_path:
            os.makedirs(self.save_path, exist_ok=True)
            eval_logger.info(f"图像保存目录设置为: {self.save_path}")

    def make_conversation(self, problem, system_prompt, prompt_template):
        problem = prompt_template.format(Question=problem)
        prompt = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": problem,
            },
        ]
        return prompt

    def extract_responses_list(
        self, 
        tokenizer, 
        input_ids: torch.Tensor,
        multi_turn_response_mask: torch.Tensor # 0,0,0,...,1,1,1,...,0,0,0,...,1,1,1
    ) -> list:
        # Tensor Method
        diff = torch.diff(multi_turn_response_mask, prepend=torch.tensor([0], device=multi_turn_response_mask.device))
        starts = torch.where(diff == 1)[0]
        mask_appended = torch.cat([multi_turn_response_mask, torch.tensor([0], device=multi_turn_response_mask.device)], dim=0)
        diff_end = torch.diff(mask_appended)
        ends = torch.where(diff_end == -1)[0] - 1
        segments = []
        for s, e in zip(starts, ends):
            segments.append(input_ids[s:e+1].tolist())

        # Decode each segment
        # decoded_responses = [tokenizer.decode(seg, skip_special_tokens=True) for seg in segments]
        decoded_responses = tokenizer.batch_decode(segments, skip_special_tokens=True)
        return decoded_responses

    # Function to encode the image
    def encode_image(self, image: Union[Image.Image, str], downsample_image=False, ds_factor=2):
        if isinstance(image, str):
            img = Image.open(image).convert("RGB")
        else:
            img = image.copy()

        # Ensure minimum dimensions of 28x28
        width, height = img.size
        if downsample_image:
            new_width = width // ds_factor
            new_height = height // ds_factor
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        width, height = img.size
        if width < 28 or height < 28:
            scale = max(28 / width, 28 / height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
        height,width = smart_resize(height=img.height, width=img.width, factor=28, min_pixels=4*28*28, max_pixels=2048*2048)
        img = img.resize((width, height), resample=Image.Resampling.LANCZOS)
        return img
    
    

    # Function to encode the video
    def encode_video(self, video_path, max_frames_num=8):
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frame_num = len(vr)
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int)

        # Ensure the last frame is included
        if total_frame_num - 1 not in uniform_sampled_frames:
            uniform_sampled_frames = np.append(uniform_sampled_frames, total_frame_num - 1)

        frame_idx = uniform_sampled_frames.tolist()
        frames = vr.get_batch(frame_idx).asnumpy()

        base64_frames = []
        for frame in frames:
            img = Image.fromarray(frame)
            output_buffer = BytesIO()
            img.save(output_buffer, format="PNG")
            byte_data = output_buffer.getvalue()
            base64_str = base64.b64encode(byte_data).decode("utf-8")
            base64_frames.append(base64_str)

        return base64_frames

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def generate_until(self, requests) -> List[str]:
        eval_logger.info(f"开始处理数据集，总共 {len(requests)} 个请求")
        
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        batch_size = self.batch_size_per_gpu
        batched_requests = [requests[i : i + batch_size] for i in range(0, len(requests), batch_size)]

        if self.prompt == 'tool_call':
            system_prompt = IMAGE_CROP_SYSTEM_PROMPT
            prompt_template = "Answer the question based on the image provided. You must conduct reasoning within <think> and </think> first in each of your reasoning steps. You may call ONE function tool per step to help you better solve the problem. Place the function tool within <tool_call> and </tool_call> at the end of each step to perform a function call. You should continue your reasoning process within <think> and </think> based on the content returned by the function tool. Once you confirm your final answer, place the final answer inside <answer> and </answer>. For mathematical or multiple-choice problem, wrap the answer value or choice with \\boxed{{}}. Here is the image and question:\n{Question}"
        elif self.prompt == 'raw_prompt':
            system_prompt = "You are a helpful assistant."
            prompt_template = "Answer the question based on the image provided. You must conduct reasoning within <think> and </think> first in each of your reasoning steps. Once you confirm your final answer, place the final answer inside <answer> and </answer>. For mathematical or multiple-choice problem, wrap the answer value or choice with \\boxed{{}}. Here is the image and question:\n{Question}"
        else:
            print(f"Invalid prompt type: {self.prompt}")
            raise NotImplementedError

        for batch_requests in batched_requests:
            batched_vllm_inputs = []
            multi_turn_response_mask = []
            prefix_prompt_lengths = []
            for idx in range(len(batch_requests)):
                contexts, gen_kwargs, doc_to_visual, doc_id, task, split = batch_requests[idx].arguments
                if "max_new_tokens" not in gen_kwargs:
                    gen_kwargs["max_new_tokens"] = 16384
                if "temperature" not in gen_kwargs:
                    gen_kwargs["temperature"] = 0
                if "top_p" not in gen_kwargs or gen_kwargs["top_p"] <= 0 or gen_kwargs["top_p"] >= 1:
                    gen_kwargs["top_p"] = 0.95

                params = {
                    "temperature": gen_kwargs["temperature"],
                    "max_tokens": 1600, #gen_kwargs["max_new_tokens"],
                    "top_p": gen_kwargs["top_p"],
                    "n": 1
                }
                sampling_params = vllm.SamplingParams(**params)

                visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]
                if None in visuals:
                    visuals = []
                    imgs = []
                else:
                    visuals = self.flatten(visuals)
                    imgs = []  # multiple images or frames for video
                    original_imgs = []
                    for visual in visuals:
                        if isinstance(visual, str) and (".mp4" in visual or ".avi" in visual or ".mov" in visual or ".flv" in visual or ".wmv" in visual):
                            frames = self.encode_video(visual)
                            imgs.extend(frames)
                        elif isinstance(visual, str) and (".jpg" in visual or ".jpeg" in visual or ".png" in visual or ".gif" in visual or ".bmp" in visual or ".tiff" in visual or ".webp" in visual):
                            img = self.encode_image(visual)
                            imgs.append(img)
                        elif isinstance(visual, Image.Image):
                            img = self.encode_image(visual, self.downsample_image, ds_factor=self.ds_factor)
                            original_img = self.encode_image(visual)
                            imgs.append(img)
                            original_imgs.append(original_img)
                num_images = len(imgs)
                image_tokens = "\n".join(["<image>"] * num_images)
                user_content = image_tokens + "\n" + contexts
                chat = self.make_conversation(user_content, system_prompt, prompt_template)
                prompt_with_chat_template = self.tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)
                raw_prompt = prompt_with_chat_template.replace('<image>', '<|vision_start|><|image_pad|><|vision_end|>')
                raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
                # vllm_input = {'prompt_token_ids': deepcopy(raw_prompt_ids), 'multi_modal_data': {"image": deepcopy(imgs)}}
                vllm_input = {'prompt_token_ids': deepcopy(raw_prompt_ids), 'multi_modal_data': {"image": deepcopy(imgs)}, 'original_image': original_imgs}
                
                prefix_length = len(raw_prompt_ids)
                batched_vllm_inputs.append(vllm_input)
                multi_turn_response_mask.append([torch.zeros(prefix_length)]) # [torch.Tensor(prefix_length,)]
                prefix_prompt_lengths.append(prefix_length)

            to_generate = list(range(len(batched_vllm_inputs)))
            max_image_num = self.max_images
            current_iteration = 0
            while current_iteration < self.max_generation_round and len(to_generate) > 0:
                idx_to_gen = [] # list of vllm_inputs, at first the length is B'*R
                for i in to_generate:
                    idx_to_gen.append(batched_vllm_inputs[i])
                eval_logger.info(f"[Round #{current_iteration} Rollout START] For THIS round, We hava {len(idx_to_gen)} trajs to complete ...")
                outputs = self.inference_engine.generate(
                    prompts=idx_to_gen,  # list of dict
                    sampling_params=sampling_params,
                    use_tqdm=False
                )
                response = [] # list of tuple, B'*R, valid(no-pad) response_ids with unequal length
                for output in outputs:
                    for sample_id in range(len(output.outputs)):
                        # HACK: filter > (voc_size + specidal_token_num) token_ids, 151664 for qwen model
                        _token_ids = output.outputs[sample_id].token_ids
                        filtered_token_ids = [token_id for token_id in _token_ids if token_id <= 151664]    # NOTE: <tool_call>: 151657, </tool_call>: 151658
                        if 151645 not in filtered_token_ids:
                            # replace the last token with <|im_end|> if no <|im_end|> in response,
                            # this is to ensure successful execution of get_final_eos_mask in multi-turn scenario
                            filtered_token_ids[-1] = 151645
                        response.append(filtered_token_ids)

                # attach model responses to vllm_inputs
                assert len(to_generate) == len(response)

                idx_to_remove = []
                id_tool_query_mapping = {}
                for i_gen, response_ in zip(to_generate, response):
                    # update conversation
                    response_ = list(response_)
                    batched_vllm_inputs[i_gen]['prompt_token_ids'] += response_
                    multi_turn_response_mask[i_gen].append(torch.ones(len(response_)))
                    # [TOOL CALL TRIGGER] We check model's last turn response, if not any tool called, then remove this traj from to_generate
                    decoded_resp_ = self.tokenizer.decode(response_, skip_special_tokens=True)
                    pattern = re.compile(r'<tool_call>(.*?)</tool_call>', re.DOTALL)
                    tool_call_contents = pattern.findall(decoded_resp_)
                    if len(tool_call_contents) > 0:
                        if len(batched_vllm_inputs[i_gen]['multi_modal_data']['image']) >= max_image_num:   # If the current traj has already reached max_image_num, but still try to call tool, we should remove this traj.
                            idx_to_remove.append(i_gen)
                            eval_logger.info(f"Traj {i} exceeds maximum function tool call num {len(batched_vllm_inputs[i]['multi_modal_data']['image'])}")
                        assert str(i_gen) not in id_tool_query_mapping.keys()
                        error_info = None
                        try:
                            json_pattern = re.compile(r'\{.*?\}\}')
                            json_objects = []
                            for content in tool_call_contents:
                                json_strings = json_pattern.findall(content)
                                json_objects.extend([json.loads(json_str) for json_str in json_strings])
                            tool_type, bbox_2d = prepare_bbox_tool_call_inputs(json_objects)
                        except Exception as e:
                            eval_logger.info(str(e))
                            error_info = str(e)
                            tool_type = None
                            bbox_2d = None
                        id_tool_query_mapping[str(i_gen)] = {
                            "tool_type": tool_type,
                            "error_info": error_info,
                            "decoded_response": decoded_resp_, 
                        }
                    # Direct Answer
                    else:
                        # remove this traj from to_generate
                        idx_to_remove.append(i_gen)
                        # NOTE: to_generate.remove(i_gen) # DO NOT .remove() in for loop
                    # eval_logger.info(f"[Round #{current_iteration}] i_gen: {i_gen} | resp: {self.tokenizer.decode(response_, skip_special_tokens=True)}")
                if to_generate and id_tool_query_mapping:   # Make sure to PRINT when to_generate and id_tool_query_mapping is not None
                    eval_logger.info(f"[Round #{current_iteration}] Example Generation: to_generate[0]: {to_generate[0]} | response[0]: {self.tokenizer.decode(response[0], skip_special_tokens=True)}")
                    eval_logger.info(f"[Round #{current_iteration} Rollout Tool Call Trigger] For THIS round, ids {next(iter(id_tool_query_mapping))} need to apply function tool using: {id_tool_query_mapping[next(iter(id_tool_query_mapping))]} ...")
                else:
                    eval_logger.info(f"[Round #{current_iteration} Rollout Tool Call Trigger] No ids need to apply function tool for this round.")
                # update 'to_generate'
                for x in idx_to_remove:
                    to_generate.remove(x)  

                eval_logger.info(f"[Round #{current_iteration} Rollout END] For NEXT round, We hava {len(to_generate)} trajs to complete ...")

                function_tool_results = []
                bbox_tasks = [] 
                
                
                for i_todo in to_generate:
                    assert str(i_todo) in id_tool_query_mapping.keys()
                    image_input = batched_vllm_inputs[i_todo]['multi_modal_data']['image'][0]
                    image_to_crop = batched_vllm_inputs[i_todo]['original_image'][0]
                    tool_type = id_tool_query_mapping[str(i_todo)]['tool_type']
                    error_info = id_tool_query_mapping[str(i_todo)]["error_info"]
                    decoded_response = id_tool_query_mapping[str(i_todo)]["decoded_response"]
                    
                    if tool_type == "request_local_region":
                        bbox_tasks.append({
                            'i_todo': i_todo,
                            'image_input': image_input,
                            'image_to_crop': image_to_crop,
                            'decoded_response': decoded_response
                        })
                    else:
                        function_tool_results.append((i_todo, error_info))
                
                if bbox_tasks:
                    eval_logger.info(f"开始批量处理 {len(bbox_tasks)} 个bbox任务")
                    
                    # 并行解析bbox
                    def process_single_bbox_task(task):
                        image_input = task['image_input']
                        image_to_crop = task['image_to_crop']
                        decoded_response = task['decoded_response']
                        i_todo = task['i_todo']
                        
                        # 使用缓存优化bbox解析
                        cache_key = hash(decoded_response + str(image_input.size))
                        if cache_key in self.bbox_cache:
                            bbox = self.bbox_cache[cache_key]
                        else:
                            bbox = get_bbox_from_response(decoded_response, image_input, ds_factor=self.ds_factor)
                            if bbox:
                                self.bbox_cache[cache_key] = bbox
                        
                        if bbox:
                            img_cache_key = hash(str(image_to_crop.tobytes()) + str(bbox))
                            if img_cache_key in self.image_cache:
                                tool_outputs = self.image_cache[img_cache_key]
                            else:
                                tool_outputs = crop_image_by_bbox(image_to_crop, bbox)
                                if len(self.image_cache) < 100:
                                    self.image_cache[img_cache_key] = tool_outputs
                        else:
                            tool_outputs = "Error: Failed to parse bbox from response"
                        
                        return (i_todo, tool_outputs)
                    
                    with ThreadPoolExecutor(max_workers=min(4, len(bbox_tasks))) as executor:
                        bbox_results = list(executor.map(process_single_bbox_task, bbox_tasks))
                    
                    function_tool_results.extend(bbox_results)
                    
                
                result_dict = {i_todo: result for i_todo, result in function_tool_results}
                function_tool_results = [result_dict[i_todo] for i_todo in to_generate]

                # [Process Tool Call Results]
                to_generate_ = to_generate.copy() # make a copy since we will be modifying to_generate
                assert len(to_generate_) == len(function_tool_results)

                for i_gen_, tool_call_result_ in zip(to_generate_, function_tool_results):
                    if isinstance(tool_call_result_, Image.Image):
                        tool_call_prompt_message = "<|im_start|>user\n" + "<tool_response>\nThe cropped region of the image is shown below:\n<|vision_start|><|image_pad|><|vision_end|>\n</tool_response>\n" + IMAGE_CROP_TOOL_CALL_MULTI_TRUN_PROMPT + "<|im_end|>\n<|im_start|>assistant\n"
                        next_turn_prompt_ids = self.tokenizer.encode(tool_call_prompt_message)
                        # update conversation
                        batched_vllm_inputs[i_gen_]['prompt_token_ids'] += next_turn_prompt_ids # this might go over response length, but we will cut it later by 'max_total_response_length'
                        batched_vllm_inputs[i_gen_]['multi_modal_data']['image'].append(tool_call_result_)
                        multi_turn_response_mask[i_gen_].append(torch.zeros(len(next_turn_prompt_ids)))
                    else:
                        tool_call_prompt_message = "<|im_start|>user\n" + tool_call_result_ + ERROR_INFO_MULTI_TURN_PROMPT + "<|im_end|>\n<|im_start|>assistant\n"
                        next_turn_prompt_ids = self.tokenizer.encode(tool_call_prompt_message)
                        batched_vllm_inputs[i_gen_]['prompt_token_ids'] += next_turn_prompt_ids # this might go over response length, but we will cut it later by 'max_total_response_length'
                        multi_turn_response_mask[i_gen_].append(torch.zeros(len(next_turn_prompt_ids), dtype=torch.int64))
                # update iteration count
                current_iteration += 1
            
            # assert self.enable_tool_call
            response_text = []
            for i_ in range(len(batched_vllm_inputs)):
                first_round_prompt_length = prefix_prompt_lengths[i_]
                generation_response_ids = batched_vllm_inputs[i_]['prompt_token_ids'][first_round_prompt_length:]
                generation_response_masks = torch.cat(multi_turn_response_mask[i_][1:], dim=0)
                valid_indices = generation_response_masks.nonzero(as_tuple=True)[0]
                valid_generation_response_ids = [generation_response_ids[i] for i in valid_indices.tolist()]
                generation_text = self.tokenizer.decode(valid_generation_response_ids, skip_special_tokens=True)
                response_text.append(generation_text)
                
                if self.save_path and 'original_image' in batched_vllm_inputs[i_] and batched_vllm_inputs[i_]['original_image']:
                    contexts, gen_kwargs, doc_to_visual, doc_id, task, split = batch_requests[i_].arguments
                    task_save_dir = os.path.join(self.save_path, task, split)
                    os.makedirs(task_save_dir, exist_ok=True)
                    
                    original_imgs = batched_vllm_inputs[i_]['original_image']
                    for visual_idx, original_img in enumerate(original_imgs):
                        if len(original_imgs) == 1:
                            save_filename = f"{doc_id}.png"
                        else:
                            save_filename = f"{doc_id}_image_{visual_idx}.png"
                        
                        save_filepath = os.path.join(task_save_dir, save_filename)
                        
                        # 直接保存原图
                        original_img.save(save_filepath)
                        eval_logger.info(f"已保存原图到: {save_filepath}")

            assert len(response_text) == len(batch_requests)
            res.extend(response_text)
            pbar.update(len(batch_requests))

        pbar.close()
        
        
            
        if self.save_path:
            eval_logger.info(f"所有原图已保存到: {self.save_path}")
        
        # 清理缓存以释放内存
        if hasattr(self, 'bbox_cache'):
            cache_size = len(self.bbox_cache)
            self.bbox_cache.clear()
            eval_logger.info(f"已清理bbox缓存 ({cache_size}个条目)")
        
        if hasattr(self, 'image_cache'):
            cache_size = len(self.image_cache)
            self.image_cache.clear()
            eval_logger.info(f"已清理图像缓存 ({cache_size}个条目)")
        
        return res

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        # TODO
        assert False, "GPT4V not support"

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation")
    
    def __del__(self):
        """清理资源"""
        if hasattr(self, 'thread_pool'):
            self.thread_pool.shutdown(wait=True)
            eval_logger.info("线程池已关闭")

