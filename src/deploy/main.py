import os
import modal
import torch
from transformers import AutoModelForSeq2SeqLM, T5Tokenizer

MODEL_DIR = "/model"

image = modal.Image.debian_slim().pip_install("torch",
                                              "transformers",
                                              "sentencepiece",
                                              "huggingface")

app = modal.App("g5", image=image)

@app.cls(gpu="A10G", image=image, enable_memory_snapshot=True)
class G5Model:
    # download and store weights
    @modal.build()
    def download_model_to_folder(self):
        from huggingface_hub import snapshot_download

        os.makedirs(MODEL_DIR, exist_ok=True)
        snapshot_download("zongleon/g5_human_mouse_translation", local_dir=MODEL_DIR)
    
    # on contained load, load weights
    @modal.enter(snap=True)
    def setup(self):
        self.tokenizer = T5Tokenizer.from_pretrained(MODEL_DIR)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR,
            torch_dtype=torch.float16
        ).to("cuda")

        # compile
        if torch.__version__ >= "2.0":
            self.model = torch.compile(self.model)

    # inference endpoint
    @modal.web_endpoint()
    def inference(self, sequence):
        inputs = self.tokenizer(" ".join(sequence), return_tensors="pt").input_ids.to("cuda")
        outputs = self.model.generate(inputs, max_new_tokens=len(sequence))
        pred_seq = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return pred_seq.replace(" ", "")
