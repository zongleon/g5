import time
from tqdm import tqdm
import requests
from urllib.error import HTTPError
import pandas as pd

df = pd.read_csv("/scratch/lzong/projects/g5/data/prot_mart_export.csv")

df = df.dropna(subset=["Protein stable ID", "Mouse protein or transcript stable ID"])

req_count = 0
last_req = 0
    
def fetch_ensembl_sequence(ensembl_id, seq_type="protein"):
    global req_count
    global last_req
    url = f"https://rest.ensembl.org/sequence/id/{ensembl_id}?"
    headers = {"Content-Type": "text/plain"}
    params = {"type": seq_type}
    if req_count >= 15:
        delta = time.time() - last_req
        if delta < 1:
            time.sleep(1 - delta)
        last_req = time.time()
        req_count = 0
        
    try:
        response = requests.get(url, headers=headers, params=params)
        req_count += 1
        if response.ok:
            return response.text
        else:
            return None
        
    except Exception as e:
        if hasattr(e, "code") and e.code == 429:
            if hasattr(e, "headers") and 'Retry-After' in e.headers:
                retry = e.headers['Retry-After']
                time.sleep(float(retry))
                return fetch_ensembl_sequence(ensembl_id)
        return None

seqs = []
for ensembl_id in tqdm(df["Protein stable ID"]):
    seqs.append(fetch_ensembl_sequence(ensembl_id))
    
df["hum_seq"] = seqs

df.to_csv("/scratch/lzong/projects/g5/data/prot_only_human.csv", index=False)

# Fetch and save sequences
seqs = []
for ensembl_id in tqdm(df["Mouse protein or transcript stable ID"]):
    seqs.append(fetch_ensembl_sequence(ensembl_id))

df["mouse_seq"] = seqs
            
df.to_csv("/scratch/lzong/projects/g5/data/prot_sequences.csv", index=False)