from transformers import (
    AutoTokenizer,
)
from datasets import load_dataset
from itertools import chain

dataset = load_dataset("text", 
                       data_files={"train": "data/train.txt", 
                                   "dev": "data/dev.txt"},
                      cache_dir="data/cache")

tokenizer = AutoTokenizer.from_pretrained("g5_tokenizer")

print(f"tokenizer is fast: {tokenizer.is_fast}")

def tokenize(example):
    tokenized_example = tokenizer(
       example["text"], return_special_tokens_mask=True
    )
    return tokenized_example

tokenized_ds = dataset.map(tokenize, remove_columns=["text"], batched=True)

max_len = 1110 # This number is to have an actual input size of 1000 for the model

# Main data processing function that will concatenate all texts from our dataset and generate chunks of
# max_seq_length.
def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We add a little padding so these tokens can be evenly split into examples with max_len # of tokens.
    if total_length >= max_len:
        remainder  = total_length - (total_length // max_len) * max_len
        if remainder > 0:
            concatenated_examples["input_ids"] += [tokenizer.pad_token_id]*(max_len - remainder)
            concatenated_examples["special_tokens_mask"] += [1]*(max_len - remainder)
            concatenated_examples["attention_mask"] += [0]*(max_len - remainder)
            if "token_type_ids" in concatenated_examples:
                # token_type_ids is 0 - we don't support next-sentence-prediction.
                concatenated_examples["token_type_ids"] += [0]*(max_len - remainder)
            total_length = len(concatenated_examples[list(examples.keys())[0]])
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + max_len] for i in range(0, total_length, max_len)]
        for k, t in concatenated_examples.items()
    }
    return result

# Note that because the batch size is 1000, the fraction of examples with pad tokens will only be <= 1/1000.
# The rest of the examples will have a full max_len tokens without padding.
tokenized_ds = tokenized_ds.map(group_texts, batched=True, num_proc=24)

print(f"the dataset contains in total {len(tokenized_ds["train"])*max_len} tokens")

tokenized_ds.save_to_disk("g5_dataset")