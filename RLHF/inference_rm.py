# load the rm model with the lora adapter applied to the sft model
# make inference on a dataset of choice and get the attributesi  
# swap the attributes back to the prompt, and 
# make inference on the same dataset for generating the output to train a attribute-conditioned model
import json
import gc
import glob
from itertools import chain
import logging
import os
import pathlib
import random
import re
from typing import Callable, Dict, List, Optional, Tuple, Union
import datasets
from dataclasses import dataclass, field
from typing import Optional, List
import accelerate
import pandas as pd
import torch
from tqdm import tqdm as tqdm_function
import transformers
import pandas as pd
import pdb
import torch

# your PyTorch code here

# clean up the cache
torch.cuda.empty_cache()
import pdb
import json
import gc
import glob
from itertools import chain
import logging
import os
import pathlib
import random
import re
from typing import Callable, Dict, List, Optional, Tuple, Union

import accelerate
import pandas as pd
import torch
import tqdm
import transformers

from peft.utils import WEIGHTS_NAME, get_peft_model_state_dict
from torch.utils.data import DataLoader, TensorDataset

from llava.constants import (
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)

from data_utils.data_utils_ppo import QueryResponseDataset

import data_utils.common_utils as common_utils

from data_utils.constants import AnswerType, FACTUAL_PROMPT

import models.rl_models as rl_models

from models.qlora_model import load_4bit_model_for_inference
from models.reward_model import load_4bit_reward_model_for_inference
from models.rl_trainer import (
    AlpacaAccelerator,
    RLTrainer,
    remove_image_token,
    truncate_after_eos_with_padding,
)
from models.ppo_trainer import remove_pad_and_left_pad

from accelerate import DistributedDataParallelKwargs

import data_utils.common_utils as utils
logger = logging.getLogger(__name__)
from llava import conversation as conversation_lib

from peft.utils import WEIGHTS_NAME, get_peft_model_state_dict
try:
    from transformers import LlamaTokenizerFast as LlamaTokenizer

    print("Using fast tokenizer")
except:
    from transformers import LlamaTokenizer

    print("Using slow tokenizer")

from transformers import AutoTokenizer, AutoModelForCausalLM

from llava.constants import (
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)

from data_utils.data_utils_ppo import QueryResponseDataset

import data_utils.common_utils as common_utils

from data_utils.constants import AnswerType, FACTUAL_PROMPT

import models.rl_models as rl_models

from models.qlora_model import load_4bit_model_for_inference
from models.reward_model import load_4bit_reward_model_for_inference
from models.rl_trainer import (
    AlpacaAccelerator,
    RLTrainer,
    remove_image_token,
    truncate_after_eos_with_padding,
)
from finetune_lora_ppo import ModelArguments, TrainingArguments, DataArguments
from data_utils.data_utils_ppo import QueryResponseDataset, QueryCoupledResponseDataset
from data_utils.data_utils_ppo import DataCollatorForQueryResponseDataset
class DisableLogger:
    def __enter__(self):
        logging.disable(logging.CRITICAL)

    def __exit__(self, exit_type, exit_value, exit_traceback):
        logging.disable(logging.NOTSET)

rm_path = '/home/ubuntu/RLHF/LLaVA-RLHF-13b-v1.5-336/rm_lora_adapter_model'
rm_model = load_4bit_reward_model_for_inference(rm_path)

vision_tower = rm_model.backbone_model.get_vision_tower()
if not vision_tower.is_loaded:
    vision_tower.load_model()
vision_tower.to(device="cuda", dtype=torch.bfloat16)
vision_tower.requires_grad_(False)
# Tokenizer
tokenizer_model_name = "/home/ubuntu/RLHF/LLaVA-RLHF-13b-v1.5-336/sft_model"
TokenizerClass = AutoTokenizer

tokenizer = TokenizerClass.from_pretrained(
    tokenizer_model_name,
    # cache_dir=args.cache_dir,
    model_max_length=2048, # taken from train_rl_model.sh
    padding_side="left",
    truncation_side="right",
    use_fast=False,
)

tokenizer.pad_token = tokenizer.unk_token
# Dataset loading



model_args = ModelArguments()
training_args = TrainingArguments()
data_args = DataArguments()
data_args.dataset_path = '/home/ubuntu/RLHF/LLaVA-RLHF-Data/llava_ppo50k-aokvqa12k-vqa10k.json'
train_instructions = datasets.load_dataset(
            "json", data_files=data_args.dataset_path
        )
train_df = pd.DataFrame(train_instructions['train'])
# Override the base model name model_args.base_model_name

model_args.base_model_name = tokenizer_model_name

model_args.vision_tower = "openai/clip-vit-large-patch14-336"

if model_args.vision_tower is not None:
    from llava.model import LlavaLlamaForCausalLM

    with DisableLogger():
        base_model = LlavaLlamaForCausalLM.from_pretrained(
            model_args.base_model_name,
            cache_dir=training_args.cache_dir,
        )

    vision_tower = base_model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model()

    data_args.image_processor = vision_tower.image_processor
    del base_model

if model_args.reward_base_model_name is None:
    model_args.reward_base_model_name = model_args.base_model_name
    data_args.reward_image_processor = vision_tower.image_processor
else:
    with DisableLogger():
        reward_base_model = LlavaLlamaForCausalLM.from_pretrained(
            model_args.reward_base_model_name,
            cache_dir=training_args.cache_dir,
        )
    reward_vision_tower = reward_base_model.get_vision_tower()
    if not reward_vision_tower.is_loaded:
        reward_vision_tower.load_model()
    data_args.reward_image_processor = reward_vision_tower.image_processor
    del reward_base_model

data_args.is_multimodal = True
data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
training_args.use_im_start_end = model_args.mm_use_im_start_end
model_args.mm_vision_select_layer = -2

### A couple critical parameter set borrowed fromt the launch script
data_args.image_aspect_ratio = 'pad'
data_args.image_folder = '/home/ubuntu/latest_llava/llava_1dot5data/coco/train2017'
training_args.query_len = 1280
training_args.response_len = 768
data_args.mm_use_im_start_end = False

############## end of arguments loading

if model_args.version in conversation_lib.conv_templates:
    conversation_lib.default_conversation = conversation_lib.conv_templates[
        model_args.version
    ]
else:
    conversation_lib.default_conversation = conversation_lib.conv_templates[
        "vicuna_v1"
    ]
train_dataset = QueryCoupledResponseDataset(
    df=train_df,
    tokenizer=tokenizer,
    query_len=training_args.query_len,
    response_len=training_args.response_len,
    data_args=data_args,
)
train_dataset.query_attn_masks = train_dataset.queries.ne(tokenizer.pad_token_id).long()

make_rl_data_module_output_dict = dict(
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=DataCollatorForQueryResponseDataset(),
    )

### Definition the inference class

class GetAttributeModel:
    def __init__(self, rm_model, tokenizer, accelerator: AlpacaAccelerator, training_args) -> None:
        self.reward_model = rm_model
        self.tokenizer = tokenizer
        self.accelerator = accelerator
        self.training_args = training_args
        self.args = training_args # TODO add other args
        self.args.reward_prompt_file = "/home/ubuntu/RLHF/LLaVA-RLHF/RLHF/scripts/13b-v1.5-336/train_reward_model.sh"
        self.train_dataset = make_rl_data_module_output_dict['train_dataset']
        self.data_collator = make_rl_data_module_output_dict['data_collator']


        self.reward_model_prompt = None
        self.reward_model_prompt_untokenized = None

        if self.args.reward_prompt_file is not None:
            with open(self.args.reward_prompt_file, "r") as f:
                self.reward_model_prompt_untokenized = " " + f.read().strip()
            self.reward_model_prompt = self.tokenizer.encode(
                self.reward_model_prompt_untokenized,
                return_tensors="pt",
                add_special_tokens=False,
            )
        self.image_to_caption_mapping = None

    def get_train_dataloader(self):
        logger.warning(
            f"Train dataset size: {len(self.train_dataset)}",
            # main_process_only=True
        )  # noqa
        train_dataloader = DataLoader(
            dataset=self.train_dataset,
            collate_fn=self.data_collator,
            batch_size=self.training_args.rollout_per_device_batch_size,
            shuffle=False,
            drop_last=False,
        )
        train_dataloader = self.accelerator.prepare(train_dataloader)  # noqa
        # self._log_batch_size(train_dataloader, "train_dataloader")
        return utils.InfiniteLoader(train_dataloader)

    def step(self, train_dataloader, step_idx: int):
        # TODO fix the range
        queries_batches = [
            next(train_dataloader) for _ in range(1)
        ]
        rollouts = self.rollout(queries_batches)
        return rollouts


    @torch.inference_mode()
    def rollout(self, queries_data) -> Dict[str, torch.Tensor]:

        """Rollout trajectories with policy.

        Args:
            queries_data: Sequence of batches or DataLoader.
                Each batch is a dict with keys 'queries' and 'query_attn_masks' and 'response'.

        Returns:
            Dictionary with keys
                'queries', 'query_attn_masks', 'responses',
                'logprobs', 'ref_logprobs', 'values',
                'rewards', 'non_score_rewards', 'shaped_rewards'.
        """
        # Give up dropout throughout.
        # self.policy.eval()
        # `keep_fp32_wrapper` retains the autocast wrapper of model.forward created by accelerate:
        #  recall one sets mixed precision options with accelerator.
        # The precise value of this arg doesn't matter here, since we use the unwrapped model only for respond.
        # Generally, try to use the wrapped model as much as you can, since it's got the autocast/cast-back wrappers.


        self.reward_model.eval()

        rollouts = []
        for batch_idx, batch in tqdm.tqdm(
            enumerate(queries_data),
            total=len(queries_data),
            disable=not self.accelerator.is_main_process,
            desc="rollout",
        ):
            gc.collect()
            torch.cuda.empty_cache()
            # Sample rollouts.
            (
                indexes,
                images,
                reward_images,
                image_file_ids,
                caption_types,
                length_bonus_multiplier,
                queries,
                query_attn_masks,
                responses,
            ) = common_utils.unpack_dict(
                common_utils.prepare_inputs(batch, device=self.accelerator.device),
                keys=(
                    "indexes",
                    "images",
                    "reward_images",
                    "image_file_ids",
                    "caption_types",
                    "length_bonus_multiplier",
                    "queries",
                    "query_attn_masks",
                    "responses"
                ),
            )

            if self.args.bf16:
                images = images.to(torch.bfloat16)
                reward_images = reward_images.to(torch.bfloat16)
            elif self.args.fp16:
                images = images.half()
                reward_images = reward_images.half()
            # TODO: replace with the response from the dataset, not model generated
            # respond_outputs = unwrapped_policy.respond(
            #     queries, query_attn_masks, images, temperature=self.args.temperature
            # )
            # (responses,) = common_utils.unpack_dict(respond_outputs, ("responses",))

            additional_token1 = self.tokenizer.encode("?", add_special_tokens=False)[0]
            assert additional_token1 == 1577

            additional_token2 = self.tokenizer.encode("\n?")[-1]
            assert additional_token2 == 29973

            responses = truncate_after_eos_with_padding(
                responses,
                self.tokenizer.eos_token_id,
                self.tokenizer.pad_token_id,
                additional_tokens=[additional_token1, additional_token2],
            )

            rollouts_batch = {
                "indexes": indexes,
                "images": images,
                "image_file_ids": image_file_ids,
                "reward_images": reward_images,
                "queries": queries,
                "query_attn_masks": query_attn_masks,
                "responses": responses,
            }
            # Decode the response 
            text_responses = self.tokenizer.batch_decode(
                responses,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            print('text responses', text_responses)
            

            correct_bonus = []
            # go through the text responses and assign bonus (optional)
            # for idx, response in enumerate(text_responses):
            #     caption_type = AnswerType(caption_types[idx].item())

            #     if caption_type == AnswerType.GENERAL:
            #         correct_bonus.append(0.0)
            #     elif caption_type in [AnswerType.A_IN_ABCD, AnswerType.B_IN_ABCD, AnswerType.C_IN_ABCD, AnswerType.D_IN_ABCD]:
            #         expected_start = caption_type.name.split("_")[0] + "."
            #         expected_phrase = "correct option is " + expected_start
            #         if response.strip().startswith(expected_start) or expected_phrase in response:
            #             correct_bonus.append(1.0)
            #         else:
            #             correct_bonus.append(0.0)
            #     elif caption_type == AnswerType.NO_IN_YESNO:
            #         if response.strip().startswith("No"):
            #             correct_bonus.append(0.5)
            #         elif response.strip().startswith("Yes"):
            #             correct_bonus.append(-0.5)
            #         else:
            #             correct_bonus.append(0.0)
            #     elif caption_type == AnswerType.YES_IN_YESNO:
            #         # TODO(zhiqings): for now, we do not give symbolic award for "Yes" in Yes/No questions.
            #         correct_bonus.append(0.0)
            #     else:
            #         raise NotImplementedError
            # assert len(correct_bonus) == len(text_responses)
            # correct_bonus = torch.tensor(correct_bonus, device=responses.device)

            has_stop_token = [
                self.tokenizer.eos_token_id in response
                for response in responses.tolist()
            ]

            # sequences = [
            #     torch.concat((query, response), dim=0)
            #     for query, response in zip(queries, responses)
            # ]
            # sequences = torch.stack(sequences, dim=0)

            sequences = remove_pad_and_left_pad(
                queries,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            # print('Text string before adding the reward model prompt', self.tokenizer.batch_decode(
            #     sequences.detach().clone(),
            #     skip_special_tokens=True,
            #     clean_up_tokenization_spaces=False,
            # )
            #)
            # Prepareing the reward model prompt, 
            # may have fancy way to add captions or context to the prompt, dubbed as FACTUAL-RLHF in LLaVa-RLHF paper
            if self.reward_model_prompt is not None:
                if self.image_to_caption_mapping is None:
                    reward_model_prompt = (
                        self.reward_model_prompt.reshape(1, -1)
                        .repeat(len(sequences), 1)
                        .to(self.accelerator.device)
                    )
                    sequences = torch.cat((sequences, reward_model_prompt), dim=1)
                else:
                    reward_model_prompt_untokenized = (
                        self.reward_model_prompt_untokenized
                    )
                    image_to_caption_mapping = self.image_to_caption_mapping

                    image_ids = []
                    for i in range(len(sequences)):
                        image_file = str(image_file_ids[i].item()).zfill(12) + ".jpg"
                        caption_type = AnswerType(caption_types[i].item())
                        if caption_type in [AnswerType.GENERAL, AnswerType.NO_IN_YESNO, AnswerType.YES_IN_YESNO]:
                            image_id = image_file
                        elif caption_type in [AnswerType.A_IN_ABCD, AnswerType.B_IN_ABCD, AnswerType.C_IN_ABCD, AnswerType.D_IN_ABCD]:
                            image_id = "aok_" + image_file
                        else:
                            print(caption_type)
                            print([AnswerType.GENERAL, AnswerType.NO_IN_YESNO, AnswerType.YES_IN_YESNO])
                            print([AnswerType.A_IN_ABCD, AnswerType.B_IN_ABCD, AnswerType.C_IN_ABCD, AnswerType.D_IN_ABCD])
                            raise NotImplementedError
                        image_ids.append(image_id)

                    captions = [
                        image_to_caption_mapping[image_id] for image_id in image_ids
                    ]

                    assert r"{factual_prompt}" in reward_model_prompt_untokenized

                    reward_model_prompts = []

                    for caption_list in captions:
                        caption_list = caption_list[:]
                        random.shuffle(caption_list)
                        factual_prompt = FACTUAL_PROMPT
                        for caption in caption_list:
                            factual_prompt = factual_prompt + f"  - {caption}\n"
                        reward_model_prompt_per_example = (
                            reward_model_prompt_untokenized.format(
                                factual_prompt=factual_prompt
                            )
                        )
                        reward_model_prompts.append(reward_model_prompt_per_example)
                    reward_model_prompts = self.tokenizer(
                        reward_model_prompts,
                        return_tensors="pt",
                        add_special_tokens=False,
                        padding="longest",
                    )["input_ids"]
                    reward_model_prompts = reward_model_prompts.to(
                        self.accelerator.device
                    )

                    sequences = torch.cat((sequences, reward_model_prompts), dim=1)
                    sequences = remove_pad_and_left_pad(
                        sequences,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )

            clean_sequences = sequences.detach().clone()
            clean_sequences[
                clean_sequences == IMAGE_TOKEN_INDEX
            ] = self.tokenizer.eos_token_id

            text_sequences = self.tokenizer.batch_decode(
                clean_sequences,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )

            if self.accelerator.is_main_process:
                print("=" * 20)
                print(text_sequences[0].split("<unk><s> ")[-1])
                print("=" * 20)
                image_id = image_file_ids[0].item()
                # convert int into "000000xxxxxx.jpg"
                image_id = (
                    "https://s3.us-east-1.amazonaws.com/images.cocodataset.org/train2017/"
                    + str(image_id).zfill(12)
                    + ".jpg"
                )
                print(image_id)
                print("=" * 20)
            # OPTIONAL: compute the length bonus 
            non_pad_mask = responses.ne(self.tokenizer.pad_token_id)
            non_pad_seq_len = (
                non_pad_mask.sum(dim=1).float().to(self.accelerator.device)
            )
            length_bonus = non_pad_seq_len / float(self.args.response_len)

            # convert length_bonus_multiplier to the shape, type, and device of length_bonus
            length_bonus = length_bonus * length_bonus_multiplier.to(
                length_bonus.device
            ).reshape(length_bonus.shape).to(length_bonus.dtype)

            sequences_attention_mask = sequences.ne(self.tokenizer.pad_token_id)

            # Evaluate logprobs of the samples.



            rollouts_batch["length_bonus"] = length_bonus
            rollouts_batch["correct_bonus"] = correct_bonus
            sub_batch_size = self.args.reward_model_per_device_batch_size
            batch_size_per_device = rollouts_batch["responses"].shape[0]
            if sub_batch_size is None or sub_batch_size == batch_size_per_device:
                reward_outputs = self.reward_model(
                    input_ids=sequences,
                    attention_mask=sequences_attention_mask,
                    images=reward_images,
                )
                print(reward_outputs)
            else:
                assert batch_size_per_device % sub_batch_size == 0

                reward_outputs_list = []

                for sub_batch_idx in range(batch_size_per_device // sub_batch_size):
                    idx_start = sub_batch_idx * sub_batch_size
                    idx_end = (sub_batch_idx + 1) * sub_batch_size
                    sub_batch_reward_outputs = self.reward_model(
                        input_ids=sequences[idx_start:idx_end],
                        attention_mask=sequences_attention_mask[idx_start:idx_end],
                        images=reward_images[idx_start:idx_end],
                    )
                    reward_outputs_list.append(sub_batch_reward_outputs)

                reward_outputs = common_utils.merge_dict(
                    reward_outputs_list, merge_fn=torch.cat
                )
                del reward_outputs_list
                del sub_batch_reward_outputs
            # Remove the penality for sequences that did not stop properly
            # reward_outputs = self.post_reward(
            #     reward_outputs,
            #     responses,
            #     penalize_no_stop_token=self.args.penalize_no_stop_token,
            #     relative_stop_token_penalty=self.args.relative_stop_token_penalty,
            #     has_stop_token=has_stop_token,
            # )
            rollouts_batch.update(reward_outputs)
            print(f'rollouts_batch: {rollouts_batch}')

            # Shape reward with KL penalty.
            # shape_reward_outputs = self._shape_reward(
            #     rewards=rollouts_batch["rewards"],
            #     responses=rollouts_batch["responses"],
            #     logprobs=rollouts_batch["logprobs"],
            #     ref_logprobs=rollouts_batch["ref_logprobs"],
            #     length_bonus=rollouts_batch["length_bonus"],
            #     correct_bonus=rollouts_batch["correct_bonus"],
            # )
            rollouts_batch_cpu = {
                key: value for key, value in rollouts_batch.items()
            }
            rollouts.append(rollouts_batch_cpu)

        # # Items in dict need to be of same shape.
        # rollouts = common_utils.merge_dict(rollouts, merge_fn=torch.cat)

        # # Estimating advantages outside the loop gives more samples for reward normalization.
        # advantages = self._estimate_advantage(
        #     rewards=rollouts["shaped_rewards"].to(self.accelerator.device),
        #     values=rollouts["values"].to(self.accelerator.device),
        # )
        # advantages = {key: value.cpu() for key, value in advantages.items()}
        # pdb.set_trace()
        return rollouts_batch
    
############ End of definition of the inference task ###############

if __name__ == "__main__":
    accelerator = AlpacaAccelerator(
    log_with=training_args.report_to,
    project_dir=training_args.logging_dir,
    gradient_accumulation_steps=training_args.gradient_accumulation_steps,
    even_batches=True,  # Make sure the batch size on each device is the same.
    split_batches=False,  # Don't break a batch into smaller chunks.
    step_scheduler_with_optimizer=False,  # Untie optimizer and scheduler step.
    # Value model might not use all parameters (e.g., lm-head) in the forward pass.
    kwargs_handlers=[
        DistributedDataParallelKwargs(
            find_unused_parameters=training_args.ddp_find_unused_parameters,
        )
    ],
)

    attribute_model = GetAttributeModel(rm_model, tokenizer, accelerator, training_args)
    attribute_model.training_args.rollout_per_device_batch_size = 8
    train_dataloader = attribute_model.get_train_dataloader()
    total_rewards_dict = {}
    for i in tqdm_function(range(2000)):
        queries_batches = [
            next(train_dataloader)
        ]
        rollout = attribute_model.rollout(queries_batches)
        image_file_ids = rollout['image_file_ids'].tolist()
        indexes = rollout['indexes'].tolist()
        rewards = rollout['rewards'].tolist()
        reward_dict = {i: (image_file_ids[i], rewards[i]) for i in range(len(indexes))}
        print(f'At {i} iteration, reward_dict: {reward_dict}')
        total_rewards_dict.update(reward_dict)
        with open(f'total_rewards_dict_1030.json', 'w') as f:
            json.dump(total_rewards_dict, f)