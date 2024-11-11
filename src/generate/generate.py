import argparse
from transformers import (
    AutoModelForSeq2SeqLM,
    BitsAndBytesConfig,
    T5Tokenizer,
)
import torch

MODEL_PATH = "../../g5_human_mouse_finetune_prot_v2/model/"
QUANTIZE = True

def load_model():
    tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50")
    
    if QUANTIZE: 
        nf4_config = BitsAndBytesConfig(
           load_in_4bit=True,
           bnb_4bit_quant_type="nf4",
           bnb_4bit_use_double_quant=True,
           bnb_4bit_compute_dtype=torch.bfloat16
        )
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH, quantization_config=nf4_config)
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)
    return tokenizer, model

def predict(tokenizer, model, human_seq):
    inputs = tokenizer(" ".join(human_seq), return_tensors="pt").input_ids

    outputs = model.generate(inputs, max_new_tokens=1000)

    pred_mouse_seq = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return pred_mouse_seq.replace(" ", "")


def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "sequence",
        help="Human sequence to translate to mouse sequence."
    )
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    tokenizer, model = load_model()
    print("Tokenizer and model loaded")
    print(predict(tokenizer, model, args.sequence))

if __name__ == "__main__":
    main()
