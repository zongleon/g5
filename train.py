# adapted from https://github.com/huggingface/olm-training/blob/main/train_model.py

from t5_data_collator import DataCollatorForT5MLM, compute_t5_input_and_target_lengths
from transformers import (
    T5ForConditionalGeneration,
    AutoTokenizer,
    set_seed,
    AutoConfig,
)

from transformers import Trainer, TrainingArguments
from datasets import load_dataset

import wandb

wandb.init("g5")

# set seed for reproducibility
seed = 0
set_seed(seed)

# define our hyperparameters
training_args = TrainingArguments(
    load_best_model_at_end=True,
    output_dir="g5_v1",
    # logging & evaluation strategies
    logging_strategy="steps",
    logging_steps=100,
    save_strategy="steps",
    save_steps=1000,
    report_to="wandb",
    # optimization parameters
    per_device_train_batch_size=32,
    learning_rate=1e-4,
    seed=seed,
    num_train_epochs=3,
    gradient_accumulation_steps=8,
    warmup_ratio=0.10,
    adam_beta1=0.9,
    adam_beta2=0.98,
    bf16 = True,
    adam_epsilon=1e-6,
    weight_decay=0.01,
)

# load processed dataset
train_dataset = load_dataset("g5dataset", split="train")

# load trained tokenizer
tokenizer = AutoTokenizer.from_pretrained("g5tokenizer")

# load config (for training from scratch, we call .from_config())
print("Training new model from scratch")
config = AutoConfig.from_pretrained("google/t5-v1_1-base")

# Set up the model and data collator.
train_dataset = train_dataset.remove_columns(["attention_mask", "special_tokens_mask"])
input_length = 1000
expanded_inputs_length, target_length = compute_t5_input_and_target_lengths(
    inputs_length=input_length,
    noise_density=0.15,
    mean_noise_span_length=3.0,
)
assert expanded_inputs_length == len(train_dataset[0]["input_ids"]),\
    f"""
    You have specified that the T5 input length should be {script_args.desired_t5_input_length}.
    In order to do this, the examples in the dataset need to be {expanded_inputs_length} before masking.
    But the examples in the dataset actually appear to be {len(train_dataset[0]['input_ids'])} tokens long.
    """
model = T5ForConditionalGeneration._from_config(config)
data_collator = DataCollatorForT5MLM(
    tokenizer=tokenizer,
    noise_density=0.15,
    mean_noise_span_length=3.0,
    input_length=input_length,
    target_length=target_length,
    pad_token_id=model.config.pad_token_id,
    decoder_start_token_id=model.config.decoder_start_token_id,
)

print(f"Resizing token embedding to {len(tokenizer)}")
model.resize_token_embeddings(len(tokenizer))

# Initialize our Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# train the model
trainer.train()

trainer.save_model(output_dir="g5_v1")