import dotenv
import json
import openai
import os
from dotenv import load_dotenv
from tqdm import tqdm
import base64
import requests
import json
import time
from tqdm import tqdm

load_dotenv()
openai.api_key = os.getenv('OPENAI_KEY')

json_path = '/home/ubuntu/RLHF/LLaVA-RLHF/RLHF/total_rewards_dict_1030_with_prompt_string_factual_mapping.json'
postfix_string = """
Return a json like the following: {accurate: float, helpful: float, language natural: float, concise: float} 
where float is a number between 0 and 1 that can be different in the 4 different categories
"""
json_path = '/home/ubuntu/RLHF/LLaVA-RLHF-Data/llava_ppo50k-aokvqa12k-vqa10k.json'
output_path = '/home/ubuntu/RLHF/LLaVA-RLHF/RLHF/total_rewards_dict_1106-1143pm_gpt4v_reward.json'
rate_limit_per_minute = 20
rate_limit_per_day = 10
coco_path = '/home/ubuntu/latest_llava/llava_1dot5data/coco/train2017'

def chat_with_openai(messages, model="gpt-4"):
    """
    Chats with OpenAI API using a series of messages and receives a model-generated message.
    :param messages: List of message dictionaries with 'role' and 'content'
    :param model: The model to use for the completion
    :return: The API response as a string
    """
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0,
    )
    return response

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

api_key = os.getenv('OPENAI_KEY')
def get_gpt_critiq(row):
  headers = {
      "Content-Type": "application/json",
      "Authorization": f"Bearer {api_key}"
  }
  question = row['conversations'][0]['value']
  captions = row['captions']
  response = row['conversations'][1]['value']
  captions_prompt = 'Ground Truth Caption description the image' + ' '.join(captions)
  prompt = captions_prompt + f'Question: {question}, response: {response}' + postfix_string
  image_path = coco_path + '/' + row['image']
  print(image_path)

  #base64_image = encode_image(image_path)

  payload = {
      "model": "gpt-4",
      "messages": [
        {
          "role": "user",
          "content": [
            {
              "type": "text",
              "text": prompt
            },
            # {
            #   "type": "image_url",
            #   "image_url": {
            #     "url": f"data:image/jpeg;base64,{base64_image}"
            #   }
            # }
          ]
        }
      ],
      "max_tokens": 300
  }

  response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
  return response.json()

# Load the list of dictionaries from the JSON file
with open(json_path, 'r') as file:
    list_of_dict = json.load(file)

output_dict = {}

# Rate limiting setup
requests_made = 0
start_time = time.time()

# Update each dictionary and add to the output_dict
for index in tqdm(range(len(list_of_dict))):
    if requests_made >= rate_limit_per_day:
        # If daily limit is reached, break out of the loop
        print("Daily rate limit reached, stopping the script.")
        break

    current_time = time.time()
    if requests_made >= rate_limit_per_minute:
        # If rate limit is reached, calculate the time to wait
        time_to_wait = 60 - (current_time - start_time)
        if time_to_wait > 0:
            print(f"Rate limit per minute reached, sleeping for {time_to_wait} seconds.")
            time.sleep(time_to_wait)

        # Reset the counter and start time
        requests_made = 0
        start_time = time.time()

    row = list_of_dict[index].copy()
    try:
        gpt_response = get_gpt_critiq(row)
        row['gpt4v_response'] = gpt_response
        output_dict[index] = row
        requests_made += 1  # Increment the successful request count
    except Exception as e:
        print(f"Error processing index {index}: {e}")
        # Implement exponential backoff or other error handling logic here

    # Save periodically to avoid losing all progress if the script stops
    if index % 10 == 0:
        with open(output_path, 'w') as f:
            json.dump(output_dict, f)

# Write the final output to the file
with open(output_path, 'w') as f:
    json.dump(output_dict, f)
