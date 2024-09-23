from transformers import AutoTokenizer
from tqdm import tqdm
from datasets import load_dataset
import gzip

vocab_size = 32_000

dataset = load_dataset("text", 
                       data_files={"train": "data/train.txt", 
                                   "dev": "data/dev.txt"},
                      cache_dir="data/cache")

def dataset_iter():
    for i in dataset["train"]:
        yield i["text"]

def batch_iterator(batch_size=10000):
    for i in tqdm(range(0, len(dataset), batch_size)):
        yield dataset["train"][i : i + batch_size]["text"]
        
tokenizer = AutoTokenizer.from_pretrained("google/t5-v1_1-base")

tokenizer = tokenizer.train_new_from_iterator(text_iterator=batch_iterator(), 
                                              vocab_size=vocab_size
                                             )

tokenizer.save_pretrained("g5_tokenizer")
