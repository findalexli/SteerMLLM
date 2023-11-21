# Extract attributes from the gpt4 response string
import traceback
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
from openai import OpenAI
import json
import os
import re
load_dotenv()
openai.api_key = os.getenv('OPENAI_KEY')
api_key = os.getenv('OPENAI_KEY')  
client = OpenAI(api_key=api_key)
# Define the directory path where your JSON files are located
dir_path = '/home/ubuntu/RLHF/LLaVA-RLHF-Data'
output_path = '/home/ubuntu/RLHF/LLaVA-RLHF/RLHF/extracted_attributes_1525_2125.json'
rate_limit_per_minute = 400
model = "gpt-4-1106-preview"
rate_limit_per_day = 10000
# List all files in the directory and sort them numerically based on the filename
file_pattern = re.compile(r'llava_7b_v1_preference_subsample_gpt4v_response_(\d+)_(\d+).json')
files = os.listdir(dir_path)
json_files = sorted([file for file in files if file_pattern.match(file)], key=lambda x: int(file_pattern.match(x).group(1)))
json_files= ['/home/ubuntu/RLHF/LLaVA-RLHF-Data/llava_7b_v1_preference_subsample_gpt4v_response_1525_1825.json',
         '/home/ubuntu/RLHF/LLaVA-RLHF-Data/llava_7b_v1_preference_subsample_gpt4v_response_1825_2125_300_combined.json']

print(f'Processing {json_files} files.')
# Initialize a list to store merged data
merged_data = {}

# Read each JSON file and append its content to the merged_data list
for file in json_files:
    with open(os.path.join(dir_path, file), 'r') as f:
        data = json.load(f)
        merged_data.update(data)  # Assuming each file contains a list of items


# json_path = '/home/ubuntu/RLHF/LLaVA-RLHF-Data/llava_7b_v1_preference_subsample_gpt4v_response_80_180.json'
# with open(json_path, 'r') as file:
#     list_of_dict = json.load(file)

system_message_json_extraction = """
Givien a response that contain something like the following: 
Ratings for Response A:
```json
{
  "Hallucinations": 0.4,  // Reference to rocky field is inaccurate.
  "Helpfulness": 0.8,  // Provides a good overview of possible difficulties.
  "Quality": 0.7,  // Coherent but includes an inaccurate detail about the terrain.
  "Spatial-Awareness": 0.6,  // Fails to accurately describe the terrain but gets other details right.
  "Domain-Knowledge": 0.8  // Understands general challenges involved in herding sheep.
}
```

Ratings for Response B:
```json
{
  "Hallucinations": 0.2,  // Mostly accurate, but there's speculation about the weather's effect.
  "Helpfulness": 0.9,  // Provides relevant difficulties associated with herding sheep.
  "Quality": 0.8,  // Good quality, with relevant details and only minor speculation.
  "Spatial-Awareness": 0.9,  // Accurately describes the spatial aspects of the image.
  "Domain-Knowledge": 0.8  // Shows good understanding of the task of herding.
}
There could be more variations of how the ratings appeared in the message

Please ensure that you are always returning a JSON with the following keys and values:
{
'Ratings for Response A:{
  "Hallucinations": 0.2,
  "Helpfulness": 0.8,
  "Quality": 0.7,
  "Spatial-Awareness": 0.9,
  "Domain-Knowledge": 0.6
},
'Ratings for Response B:{
  "Hallucinations": 0.2,
  "Helpfulness": 0.8,
  "Quality": 0.7,
  "Spatial-Awareness": 0.9,
  "Domain-Knowledge": 0.6
}
Note that the main keys are always {'Ratings for Response A} and {'Ratings for Response B} and the
subkeys are always {'Hallucinations}, {'Helpfulness}, {'Quality}, {'Spatial-Awareness}, {'Domain-Knowledge} and the values are always floats between 0 and 1.
"""


def use_gpt_to_extract_json(row):
    # Initialize the OpenAI client

    # Extract the user prompt from the row
    user_prompt = row['gpt4v_response']['choices'][0]['message']['content'] + 'Please extract the attibutes in the json format'

    # Create the payload for the request
    payload = {
        "model": model,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system_message_json_extraction},
            {"role": "user", "content": user_prompt}
        ]
    }

    # Send the request and get the response
    response = client.chat.completions.create(**payload)

    # Return the JSON response
    return response.choices[0].message.content

# Load the list of dictionaries from the JSON file

  
output_dict = {}

# Rate limiting setup
requests_made = 0
start_time = time.time()
# Update each dictionary and add to the output_dict
for index in tqdm(merged_data.keys()):
# for index in tqdm(range(len(list_of_dict))):
    # if requests_made >= rate_limit_per_day:
    #     # If daily limit is reached, break out of the loop
    #     print("Daily rate limit reached, stopping the script.")
    #     break

    # current_time = time.time()
    # if requests_made >= rate_limit_per_minute:
    #     # If rate limit is reached, calculate the time to wait
    #     time_to_wait = 60 - (current_time - start_time)
    #     if time_to_wait > 0:
    #         print(f"Rate limit per minute reached, sleeping for {time_to_wait} seconds.")
    #         time.sleep(time_to_wait)

    #     # Reset the counter and start time
    #     requests_made = 0
    #     start_time = time.time()

    row = merged_data[index].copy()
    try:
        gpt_response = use_gpt_to_extract_json(row)
        row['extracted_attributes'] = gpt_response
        print(row['extracted_attributes'])
        output_dict[index] = row
        requests_made += 1  # Increment the successful request count
    except Exception as e:
        # print traceback
        traceback.print_exc()
        print(f"Error processing index {index}: {e}")
        # Implement exponential backoff or other error handling logic here


print(f'How many rows are extracted: {len(output_dict)} out of {len(merged_data)}')
# Write the final output to the file
with open(output_path, 'w') as f:
    json.dump(output_dict, f)
