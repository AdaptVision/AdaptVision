import re
from mathruler.grader import extract_boxed_content, grade_answer
from math_verify import LatexExtractionConfig, parse, verify, ExprExtractionConfig
import os
from datetime import datetime
import time
import openai
import random
import json
import io
import base64
from typing import Optional
from PIL import Image, ImageDraw, ImageFont

from verl.utils.reward_score.general_qa_gpt import QUERY_PROMPT



SYSTEM_PROMPT = "**Your Role:** AI Reward Assessor\n**Your Goal:** Provide a binary reward (`1` or `0`).**Instructions:**1.  **Analyze Inputs:** You will receive a cropped image and a question.2.**Evaluate Sufficiency:** Ask yourself: 'Based ONLY on this image, can I get enough information to answer the question?'3.**Assign Reward:***If your answer is YES, the reward is `1`.*If your answer is NO (due to irrelevance, missing details, or ambiguity), the reward is `0`.4.**Strict Output:** Your final response must be 'Score: 1' or 'Score: 0'."
SYSTEM_PROMPT_NEW = """
**Your Role:** You are an AI agent that identifies relevant visual evidence.

**Your Goal:** Determine if an image CROP contains the **primary subject** of a given question.

**Your Golden Rule:** Your main task is to check for **presence**, not completeness. As long as the main object or area the question is asking about is clearly visible in the crop, it is considered relevant.

**Criteria for 'Score: 0' (Strictly Enforced):**
- The core subject of the question is completely absent from the image.
- The image is so blurry or corrupted that the subject is **unrecognizable**.
- The image shows something completely unrelated (e.g., question is about a car, image shows a tree).

**Your Task:**
Now, analyze the user-provided image and question following this exact process. Your response MUST only contain 'Score: 1' or 'Score: 0'.
"""

class GPT4VisionJudgeClient:
    """Client for interacting with GPT-4 Vision API"""

    def __init__(self, endpoint=None, api_key=None):
        self.api_key = os.getenv("AZURE_API_KEY", "YOUR_API_KEY")
        self.endpoint = os.getenv("AZURE_ENDPOINT", "YOUR_ENDPOINT")
        self.api_version = os.getenv("AZURE_API_VERSION", "2023-07-01-preview")
        self.client = openai.AzureOpenAI(
            azure_endpoint=self.endpoint,
            api_version=self.api_version,
            api_key=self.api_key,
        )

    def query(
        self, images, prompt: str, system_prompt: str = None, max_retries=3, initial_delay=3
    ) -> str:
        """Query GPT-4 Vision with an image and prompt"""
        # if images is None:
        #     return None

        data_url_list = []
        for image in images:
            data_url_list.append(
                get_image_data_url(image)
            )  # Assuming this function exists

        if system_prompt is not None:
            messages = [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": system_prompt},
                    ],
                },
            ]
        else:
            messages = []
        messages.append(
            {
                "role": "user",
                "content": [
                    # {"type": "text", "text": prompt},
                    # {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        )

        for data_url in data_url_list:
            messages[-1]["content"].append(
                {"type": "image_url", "image_url": {"url": data_url}}
            )

        messages[-1]["content"].append({"type": "text", "text": prompt})


        attempt = 0
        while attempt < max_retries:
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=messages,
                    temperature=min(0.2*attempt, 1.0),
                    max_tokens=16384,
                    timeout=120,
                )

                if "1" not in response.choices[0].message.content and '0' not in response.choices[0].message.content:
                    # print("Warning: there is no '0' nor '1' in the response: {}".format(
                    #     response.choices[0].message.content
                    # ))
                    raise ValueError("No '0' nor '1' in the response: {}".format(response.choices[0].message.content))
                return response.choices[0].message.content
            except openai.RateLimitError as e:
                print(str(e))
                time.sleep(3)
                continue
            except Exception as e:
                print("="*100)
                print(str(e))
                print("messages: ", messages)
                # INSERT_YOUR_CODE
                # 将报错时的messages保存到文件，便于后续排查
                try:
                    with open("gpt_judge_error_messages.json", "w", encoding="utf-8") as f:
                        import json
                        json.dump(messages, f, ensure_ascii=False, indent=2)
                except Exception as save_e:
                    print("保存messages时出错: ", save_e)
                print("="*100)
                # delay = initial_delay * (2**attempt) + random.uniform(
                #     0, 0.1 * initial_delay * (2**attempt)
                # )
                delay = 1
                time.sleep(delay)
            attempt += 1
        print(f"Warning: Failed after {max_retries} attempts")
        return ""

client_judge = GPT4VisionJudgeClient()

def get_image_data_url(image: Image.Image) -> str:
    """Convert PIL image to data URL for OpenAI image_url API."""
    with io.BytesIO() as buffer:
        image.save(buffer, format='PNG')
        b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return f"data:image/png;base64,{b64}"

def clamp_bbox_to_image(bbox, w: int, h: int):
    x0, y0, x1, y1 = bbox
    x0 = max(0, min(float(x0), float(w)))
    x1 = max(0, min(float(x1), float(w)))
    y0 = max(0, min(float(y0), float(h)))
    y1 = max(0, min(float(y1), float(h)))
    return [x0, y0, x1, y1]

def crop_image_by_bbox(image: Image.Image, bbox) -> Optional[Image.Image]:
    if image is None or bbox is None:
        return None
    w, h = image.size
    x0, y0, x1, y1 = clamp_bbox_to_image(bbox, w, h)
    if not (x1 > x0 and y1 > y0):
        return None
    return image.crop((int(x0), int(y0), int(x1), int(y1)))

# NEW: extract the real question text after the marker
def extract_question(prompt: str) -> str:
    pattern = re.compile(r"Here is the image and question:\s*(.*)", re.DOTALL | re.IGNORECASE)
    
    # 在 prompt 中搜索匹配项
    match = pattern.search(prompt)
    
    # 如果找到匹配项
    if match:
        question = match.group(1).strip()
        return question
    else:
        return None


def compute_bbox_score(prompt: str, bbox: list, extra_info: dict = None) -> float:
    """Use GPT-4o to judge whether the cropped bbox region is relevant to the question.
    Returns 1.0 if relevant/evident, else 0.0.
    Expects extra_info to provide an image via either:
      - extra_info['bbox_image_source']: PIL.Image
      - or extra_info['multi_modal_data']['image'][0]: PIL.Image
    """
    if bbox is None:
        print("bbox is None")
        return 0.0
    if not isinstance(bbox, list):
        return 0.0
    if bbox == [0,0,0,0]:
        return 0.0

        # 2) Get source image
    src_image = None
    if isinstance(extra_info, dict):
        img_candidate = extra_info.get('image', None)
        if isinstance(img_candidate, Image.Image):
            src_image = img_candidate
    if src_image is None:
        return 0.0


        # 3) Crop region
    cropped = crop_image_by_bbox(src_image, bbox)
    if cropped is None:
        return 0.0
    w, h = cropped.size
    if not (w>0 and h>0):
        return 0.0
    cropped_zoom_in = cropped.resize((w*2, h*2), resample=Image.Resampling.LANCZOS)
    if cropped_zoom_in is None:
        return 0.0

        # 4) Build judging prompt
    question = extract_question(prompt)
    if question is None:
        return 0.0
    judge_prompt = (
        "Given a question and a cropped image region, answer with 'Score: 1' if the cropped region provide information to answer the question, otherwise answer 'Score: 0'.\n\n"
        f"Question: {question}\n"
    )
      
        # 5) Query GPT-4o
    resp = client_judge.query(images=[cropped_zoom_in], prompt=judge_prompt, system_prompt=SYSTEM_PROMPT_NEW)
    if not isinstance(resp, str) or len(resp) == 0:
        return 0.0
        # 6) Parse response
    score = 1.0 if '1' in resp else 0.0
    return score
    

if __name__ == '__main__':
    question = "What company advertised the job listing of Technical Account Manager?" #"<image>\nHint: Please answer the question and provide the final answer at the end.\nQuestion: How many states are represented by the lightest color on the map?" #"<image>What is the output score when the first input is 4 and the second input is 5 according to the Hamlet Evaluation System shown in Figure 2?" #"<image>Who wrote this book?\nAnswer the question with a short phrase."
    predict_str = ["""<think>Since the qus 500 m.</think><tool_call>{"name": "request_local_region", "arguments": {"bbox_2d": [10, 400, 300, 500]}}</tool_call><answer>The total width of the Sydney Harbour Bridge at its base, as shown in the image, is \boxed{500 \text{ m}}.</answer>"""]
    ground_truth = "0" #"Martha White" #"china" #"$ 2 $" #"A" #"1:3" #"0.5 cm" #"0.5"
    image_path = "debug_rollout_images/Bbox_4O_Judge_AddToolCallReward_0808/round_0/traj_54/traj_54_round_0_input.png"
    input_image = Image.open(image_path)
    extra_info = {
        "acc_reward_weight": 1.0,
        "format_reward_weight": 0.5,
        "area_penalty_weight": 0.5,
        "gpt_extract_answer": True,
        "extract_answer_tags": "strict",
        "image": input_image
    }
    
    prompt = "Answer the question based on the image provided. You must conduct reasoning within <think> and </think> first in each of your reasoning steps. Place the function tool within <tool_call> and </tool_call> at the end of each step to perform a function call. You should continue your reasoning process based on the content returned by the function tool. Once you confirm your final answer, place the final answer inside <answer> and </answer>. For mathematical or multiple-choice problem, wrap the answer value or choice with \\boxed{{}}. Here is the image and question: What company advertised the job listing of Technical Account Manager?"
    print(extract_question(prompt))
