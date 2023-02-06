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

# Load ESM-2 model
model, alphabet = esm.pretrained.esm2_t48_15B_UR50D() #this model is 40GB to be loaded onto the GPU.
batch_converter = alphabet.get_batch_converter()
device = torch.device('cuda')
model = model.to(device)
model.eval()  

input_file = "./mouse_genes/Mus_musculus.GRCm39.pep.all.fa"
fasta_sequences = SeqIO.parse(open(input_file),'fasta')

for idx,fasta in tqdm(enumerate(fasta_sequences)):
    try:
        gene_name = fasta.description.split(" ")[7].split(":")[-1]
        transcript = fasta.description.split(" ")[4].split(":")[-1]
        name, sequence = fasta.id, str(fasta.seq)
        data = [("protein1", sequence)]
        batch_labels, batch_strs, batch_tokens = batch_converter(data)
        batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

        # Extract per-residue representations (on GPU)
        with torch.no_grad():
            results = model(batch_tokens.cuda(), repr_layers=[33], return_contacts=True)
            token_representations = results["representations"][33].cpu()

            # process whole embedding safterwards incase we want to use other data...
            # sequence_representations = []
            # for i, tokens_len in enumerate(batch_lens):
            #     sequence_representations.append(token_representations[i, 1 : tokens_len - 1].mean(0).numpy())

            sequence_representations = np.array(token_representations)
            print(sequence_representations.shape)
            embedding_file_name =  "./mouse_embeddings/" + gene_name + "_" + transcript +".npy"    
            np.save(embedding_file_name,sequence_representations)
                
    except Exception as e: 
        print(e)
