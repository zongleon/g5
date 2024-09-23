from biomart import BiomartServer
import time
from tqdm import tqdm

import pandas as pd
df = pd.read_csv("data/mart_with_human.csv")

# connect
server = BiomartServer("http://www.ensembl.org/biomart")
mart = server.datasets['mmusculus_gene_ensembl']

def fetch_sequences_in_batches(gene_ids, batch_size):
    sequences = {}
    for i in tqdm(range(0, len(gene_ids), batch_size)):
        batch_ids = gene_ids[i:i + batch_size]
        response = mart.search({
            'filters': {
                'ensembl_gene_id': batch_ids
            },
            'attributes': [
                'ensembl_gene_id', 'coding'
            ]
        })
        for line in response.iter_lines():
            parts = line.decode('utf-8').split("\t")
            gene_id, sequence = parts[0], parts[1]
            sequences[gene_id] = sequence
        time.sleep(0.5)
    return sequences

# Fetch sequences in batches
gene_ids = df['Mouse gene stable ID'].tolist()
sequences = fetch_sequences_in_batches(gene_ids, batch_size=250)

flipped = dict((v,k) for k,v in sequences.items())

# Add the sequences to the DataFrame
df['mouse_gene_sequence'] = df['Mouse gene stable ID'].map(flipped)

df.to_csv("data/mart_sequences.csv", index=False)