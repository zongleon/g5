# adapted from https://huggingface.co/docs/transformers/en/tasks/translation
import os
os.environ["TRANSFORMERS_CACHE"] = "/scratch/lzong/projects/g5/cache/"
os.environ["HF_HOME"] = "/scratch/lzong/projects/g5/cache/"

from transformers import (
    DataCollatorForSeq2Seq,
    T5Tokenizer,
    set_seed,
    AutoModelForSeq2SeqLM, 
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer
)
from datasets import load_dataset, load_from_disk
import Levenshtein
import numpy as np

import wandb

DATA_PATH = "../../g5_prot_translation_data_v2"
TOKENIZER_PATH = "Rostlab/prot_t5_xl_uniref50"
MODEL_PATH = "Rostlab/prot_t5_xl_uniref50"

OUTPUT_PATH = "../../g5_human_mouse_finetune_prot_v2"

if os.environ.get('LOCAL_RANK', '0') == '0':
    wandb.init("g5")

# set seed for reproducibility
seed = 0
set_seed(seed)

dataset = load_from_disk(DATA_PATH)
dataset = dataset.train_test_split(test_size=0.1)

tokenizer = T5Tokenizer.from_pretrained(TOKENIZER_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    edit_distances = [Levenshtein.distance(pred, ref) for pred, ref in zip(decoded_preds, decoded_labels)]
    
    # Calculate the average edit distance
    avg_edit_distance = np.mean(edit_distances)

    return {"edit_distance": avg_edit_distance}

# define our hyperparameters
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_PATH,
    local_rank=0,
    logging_steps=100,
    evaluation_strategy="steps",
    eval_steps=300,
    save_strategy="steps",
    save_steps=300,
    learning_rate=1e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,
    weight_decay=0.01,
    save_total_limit=4,
    num_train_epochs=1,
    predict_with_generate=True,
    bf16=True,
    ddp_find_unused_parameters=False,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    report_to="wandb" if os.environ.get('LOCAL_RANK', '0') == '0' else [],
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# train the model
trainer.train()

# Save model only on the main process
if os.environ.get('LOCAL_RANK', '0') == '0':
    trainer.save_model(output_dir=os.path.join(OUTPUT_PATH, "model"))
    wandb.finish()
