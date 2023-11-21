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



# openai.api_key = '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'
json_path = '/home/ubuntu/RLHF/LLaVA-RLHF-Data/llava_7b_v1_preference.json'
output_path = '/home/ubuntu/RLHF/LLaVA-RLHF-Data/llava_7b_v1_preference_subsample_gpt4v_response_1425_1525.json'

coco_path = '/home/ubuntu/latest_llava/llava_1dot5data/coco/train2017'
model = 'gpt-4-vision-preview'

system_message = """
We have engineered an AI assistant that specializes in facilitating image-based conversations. This AI, however, sometimes generates 'hallucinations' - these are inaccuracies not backed by the image or real-world data.

In this task, you will evaluate the AI's responses based on the conversation context, focusing on specific attributes. Each response should be assessed and assigned a float value (ranging from 0.0 to 1.0) in a key-value dictionary format for the following attributes:

Hallucinations: Degree to which the response includes factual inaccuracies or irrelevant details.
Helpfulness: The response's ability to effectively address the user's query or task.
Quality: Overall coherence, relevance, and presentation of the response.
Spatial-Awareness: Accuracy in interpreting and relating to the spatial aspects of the image.
Domain Knowledge: Depth and accuracy of subject-specific information provided.
Annotation Task

For each AI response, create a dictionary with the aforementioned attributes as keys and their corresponding float values as per your assessment.

Example:

json
Copy code
{
  "Hallucinations": 0.2,
  "Helpfulness": 0.8,
  "Quality": 0.7,
  "Spatial-Awareness": 0.9,
  "Domain Knowledge": 0.6
}
Input data:

[IMAGE]
[CONVERSATION CONTEXT]: This dictionary contains the history of the conversation. The current question, and reference answers from the last two turns.
Human Entry:

Key: "from" indicates the source of the message, here labeled as "human".
Value: "Can you describe the main features of this image for me?\n<image>" - This is the text of the user's query, asking for a description of an image.
GPT Entry:

Key: "from" with the value "gpt", identifying this entry as a response from the GPT model.
Value: The response text from the GPT model, which are reference answers to the user's query.
The dictionary can have multi-turn conversations, with the latest turn at the end of the list. 
Please pay attention to the last conversation turn and the latest response from the GPT model. The question is what we are asking the model 
under development to generate and the answer from gpt is a reference answer that may be imperfect. 

[RESPONSE A]: Second dictionary object answering the question from the latest conversational turn.
Key "from": Indicates the source of the message, in this case, labeled as "llava" which is the model genereated answer 
Key "value": Contains the content of the response
[RESPONSE B]: Third dictionary object answering the question from the latest conversational turn.
Key "from": Indicates the source of the message, in this case, labeled as "llava" which another the model genereated answer 
Key "value": Contains the content of the response
Output data example:

Let's think step by step. First provide your reasoning . Evaluate and rate both responses A and B using the attribute dictionary format. 
Your ratings will help determine the more appropriate response based on the specified attributes.
The rating output needs to be consistent as in following json example:
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

"""



def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def get_gpt_critiq(row):
  headers = {
      "Content-Type": "application/json",
      "Authorization": f"Bearer {api_key}"
  }
  conversations = row['conversations'][0]['value']
  response_1 = row['output_1']
  response_2 = row['output_2']
  user_prompt = f"""
  [CONVERSATION CONTEXT]: {conversations}, 
  [RESPONSE A]: {response_1}' 
  [RESPONSE B]: {response_2}'
  """
  image_path = coco_path + '/' + row['image']
  print(image_path)

  base64_image = encode_image(image_path)

  payload = {
      "model": model,
      "messages": [
        {
            "role": "system",
            "content": [
                {
                "type": "text",
                "text": system_message
                }
            ]
        },
        {
          "role": "user",
          "content": [
            {
              "type": "text",
              "text": user_prompt
            },
            {
              "type": "image_url",
              "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
              }
            }
          ]
        }
      ],
      "max_tokens": 800
  }

  response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
  return response.json()

# Load the list of dictionaries from the JSON file
with open(json_path, 'r') as file:
    list_of_dict = json.load(file)


def get_gpt4_comment(output_path, start_index, end_index):
    output_dict = {}


    # Update each dictionary and add to the output_dict
    for index in tqdm(range(start_index, end_index)):

        row = list_of_dict[index].copy()
        try:
            gpt_response = get_gpt_critiq(row)
            row['gpt4v_response'] = gpt_response
            print(gpt_response)
            output_dict[index] = row
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
    return output_dict


if __name__ == "__main__":
    load_dotenv()
    my_other_api = os.environ.get('myother_openai_api')
    shicheng_api = os.environ.get('shicheng_openai_api')
    
    environment_key = os.environ.get('OPENAI_KEY')
    
    api_lists = [my_other_api, shicheng_api, environment_key]
    start_index = 1825
    combined_output_path = f'/home/ubuntu/RLHF/LLaVA-RLHF-Data/llava_7b_v1_preference_subsample_gpt4v_response_{start_index}_{start_index + 300}_300_combined.json'
    combined_output = {}
    for api_key in api_lists:
      openai.api_key = api_key
      output_path = f'/home/ubuntu/RLHF/LLaVA-RLHF-Data/llava_7b_v1_preference_subsample_gpt4v_response_{start_index}_{start_index + 100}.json'
      output = get_gpt4_comment(output_path, start_index, start_index + 100)
      start_index += 100
      combined_output.update(output)
    # save combined ouptut
    with open(combined_output_path, 'w') as f:
        json.dump(combined_output, f)
