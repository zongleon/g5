# adapted from https://huggingface.co/docs/transformers/en/tasks/translation

from transformers import (
    DataCollatorForSeq2Seq,
    AutoTokenizer,
    set_seed,
    AutoModelForSeq2SeqLM, 
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer
)
from datasets import load_dataset, load_from_disk
import Levenshtein
import numpy as np

import wandb

DATA_PATH = "../../g5_translation_data"
TOKENIZER_PATH = "../../g5_tokenizer"
MODEL_PATH = "../../g5_v1/checkpoint-15000"

OUTPUT_PATH = "../../g5_human_mouse_finetune_v2"

wandb.init("g5")

# set seed for reproducibility
seed = 0
set_seed(seed)

dataset = load_from_disk("../../g5_translation_data")
dataset = dataset.train_test_split(test_size=0.1)

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
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
    eval_steps=100,
    learning_rate=3e-4,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=8,
    weight_decay=0.01,
    save_total_limit=4,
    num_train_epochs=3,
    predict_with_generate=True,
    bf16=True,
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

trainer.save_model(output_dir=OUTPUT_PATH + "/model")