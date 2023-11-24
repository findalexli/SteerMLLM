import json
import os
import pandas as pd
import pdb
import traceback
import ast
# This file comes right after using gpt 3.5 turbo to extract the attrbutes from the origional string
with open('/home/ubuntu/RLHF/LLaVA-RLHF/RLHF/extracted_attributes_1225_1425.json', 'r') as file:
    data = json.load(file)
df = pd.DataFrame.from_dict(data, orient='index')
# df = pd.read_csv('/home/ubuntu/RLHF/LLaVA-RLHF/RLHF/temp_30_1224.csv')
output_path = '/home/ubuntu/RLHF/LLaVA-RLHF/RLHF/gpt4_ratings/training_data/sft/extracted_attributes_sft_1225-1525-likert_scale1118.json'
prepare_for_attribute_modelling = False
prepare_for_question_answer_llava_inference = False
encode_labels_into_list_bool = False
use_likert_scale = True
prompt = 'You are are accurate and helpful assistant'
prompt += """
There are five attribute and their definitions
Hallucinations: Degree to which the response includes factual inaccuracies or irrelevant details.
Helpfulness: The response's ability to effectively address the user's query or task.
Quality: Overall coherence, relevance, and presentation of the response.
Spatial-Awareness: Accuracy in interpreting and relating to the spatial aspects of the image.
Domain-Knowledge: Depth and accuracy of subject-specific information provided.
"""
if not use_likert_scale:
    prompt += """Please rate the following conversation consisting of question-answer regarding the image in 0-1 scale in the following a dictionary format
  where the keys are Hallucinations, Helpfulness, Quality, Quality, Spatial-Awarenes Domain-Knowledge
  Please use 0-1 scale and 0 is the lowest and 1 is the highest. """
else:
    prompt += """Please rate the following question-answer regarding the image in 0-4 scale in the following format:
    where the keys are Hallucinations, Helpfulness, Quality, Quality, Spatial-Awarenes Domain-Knowledge
    Please use 0-4 scale and 0 is the lowest and 4 is the highest."""

prompt += 'Below is the question and answer regarding the image to be rated in terms of the above attributes'
columns_to_keep = ['id', 'image', 'conversations']
attributes = ['Hallucinations', 'Helpfulness', 'Quality', 'Spatial-Awareness', 'Domain-Knowledge']
def get_json_column(row):
    str =  row['extracted_attributes']
    json_string = str.replace('```json\n', '').replace('\n```', '').replace('\n', '').replace('\'', '\"').replace(' ', '')
    return json_string

df_edit = df.copy()
# Create an empty DataFrame with the same columns as df_edit
new_df = pd.DataFrame(columns=df_edit.columns)
new_df = []
for i in range(len(df)):
    row = df.iloc[i].to_dict()
    try:
        extracted_attributes = json.loads(get_json_column(df.iloc[i]))
        print(extracted_attributes)
        for key in extracted_attributes.keys():
            for attributes in extracted_attributes[key].keys():
                print(extracted_attributes[key][attributes])
                if 'Domain' in attributes:
                    row[f'{key}_Domain-Knowledge'] = extracted_attributes[key][attributes]
                else:
                    row[f'{key}_{attributes}'] = extracted_attributes[key][attributes]
        # Append the current row to new_df
        new_df.append(row)
    except:
        traceback.print_exc()

print(f'Length of new_df: {len(new_df)}, compared to original df: {len(df)}')

new_attributes = pd.DataFrame(new_df)


# Creating two separate DataFrames for output_1 and output_2
df_output_1 = new_attributes[columns_to_keep].copy()
df_output_2 = new_attributes[columns_to_keep].copy()

attributes = ['Hallucinations', 'Helpfulness', 'Quality', 'Spatial-Awareness', 'Domain-Knowledge']

# Adding attributes from RatingsforResponseA and RatingsforResponseB to the respective DataFrames
# pdb.set_trace()
for attr in attributes:
    df_output_1[attr] = new_attributes[f'RatingsforResponseA_{attr}']
    df_output_2[attr] = new_attributes[f'RatingsforResponseB_{attr}']

# Adding a column to distinguish between the two outputs
df_output_1['Output'] = new_attributes['output_1'].copy()
df_output_2['Output'] = new_attributes['output_2'].copy()

# Concatenating the two DataFrames
new_row_seperated = pd.concat([df_output_1, df_output_2], ignore_index=True)
print(f'Length of new_row_seperated: {len(new_row_seperated)}')
# Drop rows with any missing values
new_row_seperated.dropna(inplace=True)
# Create label encoding string
def encode_labels_into_string(row):
    likert_scale = 5
    items = []
    for key in attributes:
        value = row[key]
        if use_likert_scale:
            items.append(f'{key}:{round(value*(likert_scale-1))}')
        else:
            items.append(f'{key}:{value}')
    return ','.join(items)

def encode_labels_into_list(row):
    likert_scale = 5
    items = []
    for key in attributes:
        value = row[key]
        if use_likert_scale:
            items.append(str(round(value*(likert_scale-1))))
        else:
            items.append(str(value))
    return ','.join(items)
if encode_labels_into_list_bool:
    new_row_seperated['attributes_string'] = new_row_seperated.apply(encode_labels_into_list, axis=1)
else:
    new_row_seperated['attributes_string'] = new_row_seperated.apply(encode_labels_into_string, axis=1)

def _trim_conversation(conversation):
    # Parse the conversation string into a list of dictionaries
    if isinstance(conversation, str):
        conversation_list = ast.literal_eval(conversation)
    elif isinstance(conversation, list):
        conversation_list = conversation
    else:
        raise Exception("The conversation is not in the correct format.")
    # Keep only the last two turns of the conversation (one human and one gpt turn)
    if len(conversation_list) >= 2:
        trimmed_conversation = conversation_list[-2:]
    else:
        raise Exception("The conversation is too short to trim.")
    # return only the human turn
    trimmed_conversation[-2]['value'] = '<image>\n' + trimmed_conversation[-2]['value']
    return trimmed_conversation[-2]


attribute_start = 'Please answer the question such that your response is in the following attributes in 5-point Likert scale:'
def add_attribute_and_to_output_for_sft_dataset(row):
    human_turn = _trim_conversation(row['conversations'])
    human_turn['value'] += f'\n{attribute_start}\n{row["attributes_string"]}'
    llava_output = row['Output']
    if isinstance(llava_output, str):
        llava_output_dict = ast.literal_eval(llava_output)
    elif isinstance(llava_output, dict):
        llava_output_dict = llava_output
    llava_output_dict['from'] = 'gpt'
    return [human_turn, llava_output_dict]

def get_response(row):
    if isinstance(row['Output'], str):
        response = ast.literal_eval(row['Output'])['value']
    elif isinstance(row['Output'], dict):
        response = row['Output']['value']
    else:
        raise Exception("The response is not in the correct format.")
    return response

def add_attribute_and_to_output_for_attribute_modelling(row):
    # Question
    question= _trim_conversation(row['conversations'])['value']
    response = get_response(row)
    return [{'from': 'human', 'value': f'{prompt}\nQuestion: {question}\nAanswer: {response}'}, 
            {'from': 'gpt', 'value': row["attributes_string"]}]

if prepare_for_attribute_modelling:
    new_row_seperated['conversations'] = new_row_seperated.apply(add_attribute_and_to_output_for_attribute_modelling, axis=1)
    new_row_seperated.to_json(output_path, orient='records')
else:
    new_row_seperated['conversations'] = new_row_seperated.apply(add_attribute_and_to_output_for_sft_dataset, axis=1)
    new_row_seperated.to_json(output_path, orient='records')

if prepare_for_question_answer_llava_inference:
    new_row_seperated['text'] = new_row_seperated.apply(lambda row: _trim_conversation(row['conversations'])['value'].replace('<image>', ''), axis=1)
    new_row_seperated['question_id'] = new_row_seperated.apply(lambda row: row['id'], axis=1)
    new_row_seperated['ground_truth_answer'] = new_row_seperated.apply(get_response, axis=1)
    if os.path.exists(output_path):
        os.remove(output_path)
    new_row_seperated.to_json(output_path, orient='records', lines=True)