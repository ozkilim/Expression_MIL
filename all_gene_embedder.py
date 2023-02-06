from Bio import SeqIO
import scanpy as sc
import pandas as pd
import numpy as np
import shutil
import os 
import torch
import esm
from tqdm import tqdm
from esm import Alphabet, FastaBatchedDataset, ProteinBertModel, pretrained, MSATransformer
# go into tsb and get name,,, rename paerent folder...

# Load ESM-2 model
model, alphabet = esm.pretrained.esm2_t48_15B_UR50D() #this model is 40GB to be loaded onto the GPU.
batch_converter = alphabet.get_batch_converter()
device = torch.device('cuda')
model = model.to(device)
model.eval()  # disables dropout for deterministic results

adata = sc.read_h5ad('sc_training.h5ad')

input_file = "./all_genes/Mus_musculus.GRCm39.pep.all.fa"
fasta_sequences = SeqIO.parse(open(input_file),'fasta')


for idx,fasta in tqdm(enumerate(fasta_sequences)):
    try:
        gene_name = fasta.description.split(" ")[7].split(":")[-1]
        if gene_name in adata.var_names:

            name, sequence = fasta.id, str(fasta.seq)

            data = [("protein1", sequence)]

            batch_labels, batch_strs, batch_tokens = batch_converter(data)
            batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

            # Extract per-residue representations (on GPU)
            with torch.no_grad():
                results = model(batch_tokens.cuda(), repr_layers=[33], return_contacts=True)
                token_representations = results["representations"][33].cpu()
                sequence_representations = []
                for i, tokens_len in enumerate(batch_lens):
                    sequence_representations.append(token_representations[i, 1 : tokens_len - 1].mean(0).numpy())
                sequence_representations = np.array(sequence_representations)
                print(sequence_representations.shape)
                embedding_file_name =  "./all_embeddings/" + gene_name + "_protein_" + str(idx)+".npy"    
                np.save(embedding_file_name,sequence_representations)
                
    except Exception as e: 
        print(e)
