import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification
import numpy as np
import scipy.spatial.distance as sp
import argparse
from Bio import SeqIO

def parse_args():
    parser = argparse.ArgumentParser(description="Run inference for CaVepP53 model.")
    parser.add_argument('--checkpoint_path', type=str, required=True, help="Path to the model checkpoint.")
    parser.add_argument('--input_path', type=str, required=True, help="Path to the input CSV file.")
    parser.add_argument('--output_path', type=str, required=True, help="Path to the output CSV file.")
    return parser.parse_args()

def run_inference(checkpoint_path, input_path, output_path):
    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path, trust_remote_code=True).to(device)

    # Load tokenizer
    tokenizer = model.tokenizer

    # Load data
    data = pd.read_csv(input_path)
    TP53_pro=SeqIO.read(r"./datas/p53_protein_sequence.fasta", "fasta")
    WT_seq=str(TP53_pro.seq)
    # Initialize lists to store results
    results = []
    mutseqc = []
    wtseqc = []
    to_drop_indices = []
    for index, row in tqdm(data.iterrows(), total=data.shape[0], desc='generate mut_sequence'):
        pos = row["POS"]
        prof = row['REF']
        prol = row['ALT']
        WT_seq = WT_seq

        if pos - 1 >= len(WT_seq) or WT_seq[pos - 1] != prof:
            print(f"Error at index {index}: POS out of range or REF mismatch")  # 打印错误信息
            print(row )
            to_drop_indices.append(index)
            continue
        #get 51 window_size
        left_start = max(0, pos - 26)
        right_end = min(len(WT_seq), pos + 25)
        left_pad = "<pad>" * (26- (pos - left_start))
        right_pad = "<pad>" * (26 - (right_end - pos + 1))
        mutseq = left_pad + WT_seq[left_start:pos - 1] + prol + WT_seq[pos:right_end] + right_pad
        wtseq=left_pad + WT_seq[left_start:pos - 1] + prof + WT_seq[pos:right_end] + right_pad
        mutseqc.append(mutseq)
        wtseqc.append(wtseq)
    data['mut_seq']=mutseqc
    data['wt_seq']=wtseqc
    data = data.drop(to_drop_indices)
    esm_distance_list = []
    WT_avg_list = []
    MUT_avg_list = []

    # Process each row in the DataFrame
    for index, row in tqdm(data.iterrows(), total=len(data), desc="Predicting"):
        seq = row['mut_seq']
        inputs = tokenizer(seq, return_tensors="pt", padding=True, truncation=True, max_length=601).to(device)
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        predicted_class = logits.argmax(dim=-1).item()
        confidence = probabilities[0, predicted_class].item()
        new_row = row.copy()
        new_row['CaVepP53_label'] = predicted_class
        new_row['confidence_score'] = confidence

        WT_seq = row['wt_seq']
        MUT_seq = row['mut_seq']

        WT_seq_tokenized = tokenizer(WT_seq, padding=True, return_tensors='pt')
        WT_seq_tokenized.to(device)   
        output = model(**WT_seq_tokenized)
        WT_last_hidden_state = output.last_hidden_state
        
        MUT_seq_tokenized = tokenizer(MUT_seq, padding=True, return_tensors='pt')
        MUT_seq_tokenized.to(device)        
        MUT_output = model(**MUT_seq_tokenized)
        MUT_last_hidden_state = MUT_output.last_hidden_state
           
        WT_avg_hidden_state = torch.mean(WT_last_hidden_state, dim=1)
        MUT_avg_hidden_state = torch.mean(MUT_last_hidden_state, dim=1)
        
        with torch.no_grad():
            distance = torch.norm(WT_avg_hidden_state - MUT_avg_hidden_state)
        WT_avg_hidden_state = WT_avg_hidden_state.squeeze().detach().cpu().numpy()
        MUT_avg_hidden_state = MUT_avg_hidden_state.squeeze().detach().cpu().numpy()

        distance = distance.cpu().numpy().item()
        new_row['CaVepP53_score']=distance
        WT_avg_list.append(WT_last_hidden_state.squeeze().detach().cpu().numpy())
        MUT_avg_list.append(MUT_last_hidden_state.squeeze().detach().cpu().numpy())
        results.append(new_row)
        torch.cuda.empty_cache()

    results_df = pd.DataFrame(results)
    results_df = results_df.drop(columns=['mut_seq', 'wt_seq'])
    # Save results
    with open(output_path, mode='w', newline='') as file:
        results_df.to_csv(output_path, index=False)


if __name__ == "__main__":
    args = parse_args()
    run_inference(args.checkpoint_path, args.input_path, args.output_path)
