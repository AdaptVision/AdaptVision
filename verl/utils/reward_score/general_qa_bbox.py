import re
from mathruler.grader import extract_boxed_content, grade_answer
from math_verify import LatexExtractionConfig, parse, verify, ExprExtractionConfig
import os
from datetime import datetime
import time
import openai
import random
import json
from verl.utils.reward_score.gpt_judge_score import compute_bbox_score


SYSTEM_PROMPT = "You are an intelligent chatbot designed for evaluating the correctness of generative outputs for question-answer pairs.\nYour task is to compare the predicted answer with the correct answer and determine if they match meaningfully. Here's how you can accomplish the task:\n------\n##INSTRUCTIONS:\n- Focus on the meaningful match between the predicted answer and the correct answer.\n- Consider synonyms or paraphrases as valid matches.\n- Evaluate the correctness of the prediction compared to the answer."

QUERY_PROMPT = """I will give you a question related to an image and the following text as inputs:\n\n1. **Question Related to the Image**: {question}\n2. **Ground Truth Answer**: {ground_truth}\n3. **Model Predicted Answer**: {prediction}\n\nYour task is to evaluate the model's predicted answer against the ground truth answer, based on the context provided by the question related to the image. Consider the following criteria for evaluation:\n- **Relevance**: Does the predicted answer directly address the question posed, considering the information provided by the given question?\n- **Accuracy**: Compare the predicted answer to the ground truth answer. You need to evaluate from the following two perspectives:\n(1) If the ground truth answer is open-ended, consider whether the prediction accurately reflects the information given in the ground truth without introducing factual inaccuracies. If it does, the prediction should be considered correct.\n(2) If the ground truth answer is a definitive answer, strictly compare the model's prediction to the actual answer. Pay attention to unit conversions such as length and angle, etc. As long as the results are consistent, the model's prediction should be deemed correct.\n**Output Format**:\nYour response should include an integer score indicating the correctness of the prediction: 1 for correct and 0 for incorrect. Note that 1 means the model's prediction strictly aligns with the ground truth, while 0 means it does not.\nThe format should be \"Score: 0 or 1\""""

class GPT4VisionClient:
    """Client for interacting with GPT-4 Vision API"""

    def __init__(self, endpoint=None, api_key=None):
        self.api_key = os.getenv("AZURE_API_KEY", "YOUR_API_KEY")
        self.endpoint = os.getenv("AZURE_ENDPOINT", "https://api.cognitive.microsoft.com/sts/v1.0/issueToken")
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
                print("="*100)
                # delay = initial_delay * (2**attempt) + random.uniform(
                #     0, 0.1 * initial_delay * (2**attempt)
                # )
                delay = 1
                time.sleep(delay)
            attempt += 1
        print(f"Warning: Failed after {max_retries} attempts")
        return ""

client = GPT4VisionClient()

# Strictly parse binary score from GPT response to avoid misjudging '10' or '0.1'
def parse_binary_score(response: str):
    # Prefer the specified format: "Score: 0" or "Score: 1"
    m = re.search(r"\bScore\s*:\s*(0|1)\b", response, flags=re.IGNORECASE)
    if m:
        return int(m.group(1))
    # Fallback: find isolated 0/1 tokens not part of larger numbers
    tokens = re.findall(r"\b(0|1)\b", response)
    if tokens:
        # If multiple tokens exist, use the last one assuming final decision
        return int(tokens[-1])
    return None

def is_valid_direct_answer(response, direct_answer_format) -> bool:
    pattern = direct_answer_format
    # 1). Structure Matching
    if not re.match(pattern, response, re.DOTALL):
        return False
    # 2). Pattern Count
    if response.count('<think>') != 1 or response.count('</think>') != 1:
        return False
    if response.count('<answer>') != 1 or response.count('</answer>') != 1:
        return False
    # 3). <tool_call> </tool_call> is not allowed!
    if '<tool_call>' in response or '</tool_call>' in response:
        return False
    return True

def is_valid_second_turn_answer(response, second_turn_answer_format) -> bool:
    pattern = second_turn_answer_format
    if not re.match(pattern, response, re.DOTALL):
        return False
    if response.count('<think>') != 1 or response.count('</think>') != 1:
        return False
    if response.count('<answer>') != 1 or response.count('</answer>') != 1:
        return False
    
    #  <tool_call> </tool_call> is not allowed in the second turn answer
    if '<tool_call>' in response or '</tool_call>' in response:
        return False
    return True

def is_valid_first_turn_tool_call(response, first_turn_tool_call_format) -> bool:
    pattern = first_turn_tool_call_format
    # 1). Structure Matching
    if not re.match(pattern, response, re.DOTALL):
        return False
    # 2). <think> Count
    if response.count('<think>') != 1 or response.count('</think>') != 1:
        return False
    # 3). <tool_call> </tool_call> Count
    if response.count('<tool_call>') != 1 or response.count('</tool_call>') != 1:
        return False
    # 4). <answer> or </answer> is not allowed!
    if '<answer>' in response or '</answer>' in response:
        return False
    return True

def format_reward(predict_str_list: list, extra_info: dict):
    conv_rounds = len(predict_str_list)
    format_score, tool_call_count = 0, 0
    # All allowed formats
    direct_answer_format = r'^<think>.*</think>.*<answer>.*</answer>$'
    direct_answer_format_with_tool = r'^<think>.*</think>.*<tool_call>.*</tool_call>.*<answer>.*</answer>$'
    second_turn_answer_format = r'^<think>.*</think>.*<answer>.*</answer>$'
    first_turn_tool_call_format = r'^<think>.*</think>.*<tool_call>.*</tool_call>$'
    tool_call_pattern = re.compile(r'<tool_call>(.*?)</tool_call>', re.DOTALL)
    # 1-turn
    if conv_rounds == 1:
        response = predict_str_list[0].strip()
        tool_call_contents = tool_call_pattern.findall(response)
        if len(tool_call_contents) > 0:
            tool_call_count += 1
        # Direct Answer
        if is_valid_direct_answer(response, direct_answer_format):
            format_score = 1
    # multi-turn
    else:
        tool_call_match_flag = True
        for response in predict_str_list[:-1]:
            response = response.strip()
            tool_call_contents = tool_call_pattern.findall(response)
            if len(tool_call_contents) > 0:
                tool_call_count += 1
            # Call Function Tool
            if not is_valid_first_turn_tool_call(response, first_turn_tool_call_format):
                tool_call_match_flag = False
                break
        final_answer_match_flag = is_valid_second_turn_answer(predict_str_list[-1], second_turn_answer_format)
        if tool_call_match_flag and final_answer_match_flag:
            format_score = 1
            
    return format_score, tool_call_count

def inner_acc_reward(prompt: str, predict_str_list: list, original_answer: str, use_gpt=False, gpt_extract_answer=False, extra_info=None):
    original_predict_str = ' '.join(predict_str_list)
    if gpt_extract_answer:
        if extra_info['extract_answer_tags'] == 'split':
            original_predict_str = original_predict_str.split("<answer>")[-1].split("</answer>")[0].strip()
        elif extra_info['extract_answer_tags'] == 'strict':
            extract_answer_pattern = r'<answer>(.*?)</answer>'
            match = re.search(extract_answer_pattern, original_predict_str, re.DOTALL)
            if match:
                original_predict_str = match.group(1)
            else:
                reward = 0.0
                return reward
        else:
            raise ValueError("Such value is not implemented for extra_info['extract_answer_tags']: {}".format(extra_info['extract_answer_tags']))
    question = prompt
    prompt = QUERY_PROMPT.format(question=question, ground_truth=original_answer, prediction=original_predict_str)
    response = client.query(images=[], prompt=prompt, system_prompt=SYSTEM_PROMPT)
    if len(response) == 0:
        reward = {"is_filter": True, "info": "error with gpt4o"}
    else:
        parsed = parse_binary_score(response)
        if parsed is None:
            reward = {"is_filter": True, "info": "invalid gpt score format"}
        else:
            reward = float(parsed)
    return reward

def acc_reward(prompt: str, predict_str_list: list, solution: str, extra_info: dict = None) -> float:
    gpt_extract_answer = extra_info.get("gpt_extract_answer", False)
    reward = inner_acc_reward(prompt, predict_str_list, solution, use_gpt=True, gpt_extract_answer=gpt_extract_answer, extra_info=extra_info)
    return reward


def parse_bbox_from_predict(predict_str_list: list):
    """Parse bbox from the first response segment if a tool_call is present.
    Returns bbox list [x0,y0,x1,y1] or None.
    """
    import json, re
    if not predict_str_list:
        return None
    tool_call_pattern = r"<tool_call>(.*?)</tool_call>"
    tool_call_match = re.search(tool_call_pattern, predict_str_list[0], re.DOTALL)
    if not tool_call_match:
        return None
    tool_call_content = tool_call_match.group(1).strip()
    try:
        tool_call_data = json.loads(tool_call_content)
        if 'name' in tool_call_data and 'arguments' in tool_call_data and 'bbox_2d' in tool_call_data['arguments'] and len(tool_call_data['arguments']['bbox_2d']) == 4:
            bbox = tool_call_data["arguments"]["bbox_2d"]
        else:
            return None
    except Exception:
        return None
    
    if not isinstance(bbox, list):
        return None

    if not all(isinstance(coord, (int, float)) for coord in bbox):
        return None

    x0, y0, x1, y1 = bbox
   
    if not (x1 > x0 and y1 > y0):
        return None
    return [float(x0), float(y0), float(x1), float(y1)]

def compute_bbox_area_ratio(bbox, image_width: float, image_height: float):
    if not bbox or not image_width or not image_height:
        return None
    x0, y0, x1, y1 = bbox
    w = max(0.0, min(float(x1 - x0), float(image_width)))
    h = max(0.0, min(float(y1 - y0), float(image_height)))
    area = w * h
    total = float(image_width) * float(image_height)
    if total <= 0:
        return None
    return area / total

def compute_score(prompt: str, predict_str_list: list, ground_truth: list, extra_info: dict = None) -> float:
    acc_reward_weight = extra_info.get('acc_reward_weight', 1.0) if extra_info else 1.0
    format_reward_weight = extra_info.get('format_reward_weight', 1.0) if extra_info else 1.0
    area_penalty_weight = extra_info.get('area_penalty_weight', 0.0) if extra_info else 0.0
    acc_condition = extra_info.get('acc_condition', False) if extra_info else False
    min_area_ratio = extra_info.get('min_area_ratio', 0.0) if extra_info else 0.0 
    bbox_reward_weight = extra_info.get('bbox_reward_weight', 0.0) if extra_info else 0.0 
    tool_call_penalty = extra_info.get('tool_call_penalty', 0.0) if extra_info else 0.0
    acc = acc_reward(prompt, predict_str_list, ground_truth, extra_info)
    if isinstance(acc, dict):
        return acc
    format_score, tool_call_count = format_reward(predict_str_list, extra_info)

    acc_score = acc_reward_weight * acc
    format_score = format_reward_weight * format_score

    # BBox area ratio penalty
    area_penalty_score = 0.0
    area_ratio = 0.0
    img_w = extra_info.get('bbox_image_width', None)
    img_h = extra_info.get('bbox_image_height', None)
    if tool_call_count == 0:
        area_ratio = 0
        area_penalty_score = 0.0
        bbox_score = 0.0
        score = acc_score + format_score
        return score, acc_score, format_score, area_penalty_score, area_ratio, bbox_score, tool_call_count
    
    else:
        bbox = parse_bbox_from_predict(predict_str_list)
        if area_penalty_weight > 0.0 and img_w and img_h:
            if bbox is None:
                area_ratio = 0     
            else:
                area_ratio = compute_bbox_area_ratio(bbox, img_w, img_h)
            area_penalty_score = area_penalty_weight * max(0, area_ratio - min_area_ratio)

        bbox_score = 0
        if bbox_reward_weight > 0.0:
            if bbox is None:
                bbox_reward = 0
            else:
                bbox_reward = compute_bbox_score(prompt, bbox, extra_info)
            bbox_score = bbox_reward_weight * bbox_reward
        
        tool_penalty_factor = (1 - tool_call_penalty) if tool_call_count > 0 else 1.0
        score = tool_penalty_factor * acc_score + format_score
        
        if acc_condition and acc == 0.0:
            bbox_score = 0
            area_penalty_score = 0
            
        score += (bbox_score - area_penalty_score)
        return score, acc_score, format_score, area_penalty_score, area_ratio, bbox_score, tool_call_count

if __name__ == '__main__':
    question = "What is the total width of the Sydney Harbour Bridge at its base as shown in the image?" #"<image>\nHint: Please answer the question and provide the final answer at the end.\nQuestion: How many states are represented by the lightest color on the map?" #"<image>What is the output score when the first input is 4 and the second input is 5 according to the Hamlet Evaluation System shown in Figure 2?" #"<image>Who wrote this book?\nAnswer the question with a short phrase."
    predict_str = ["""<think>From the image, we observe a representation of the Sydney Harbour Bridge where the bottom section of the bridge and its base, which includes water, are visible. The width of the bridge's base that is on land appears to be 108m, and the width in the water appears to be 500m. To determine the total width, we need to sum these two dimensions.</think>"""]
    ground_truth = "The total width of the Sydney Harbour Bridge at its base, as shown in the image, is 503.3 meters" #"Martha White" #"china" #"$ 2 $" #"A" #"1:3" #"0.5 cm" #"0.5"
    extra_info = {
        "acc_reward_weight": 1.0,
        "format_reward_weight": 0.5,
        "area_penalty_weight": 0.5,
        "gpt_extract_answer": True,
        "extract_answer_tags": "split",
    }
    s1 = compute_score(question, predict_str, ground_truth, extra_info)
    print(s1)

    # s2 = format_reward(predict_str, extra_info)
    # print(s2)
    
    # s3 = compute_bbox_area_ratio(parse_bbox_from_predict(predict_str), 100, 100)
    # print(s3)
