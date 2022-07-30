
from transformers import BertConfig, BertForSequenceClassification, DNATokenizer
import pandas as pd
import torch
import numpy as np
import torch.nn as nn
import os
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from utils import compute_correct_attention_masks

def main():
    # INPUT
    batch_size = 1
    kmer = 6
    predict_dir = "results/"

    # Info
    model_path = "dnabert6/"

    # Load config, model and tokenizer
    tokenizer = DNATokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path, 
                                                        num_labels=5, # types of SV
                                                        output_attentions=True)

    # Select device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        n_gpu = torch.cuda.device_count()
        if n_gpu > 1:
            print(f"Using {n_gpu} GPUs!")
            model.to(device)
            model = nn.DataParallel(model)
    else:
        device = torch.device("cpu")
        model.to(device)
        n_gpu = torch.cuda.device_count()

    # Read sequences
    d_eval = pd.read_csv(f"fine_tune_data/eval.tsv", sep="\t")

    # Prepare tokenized sequences
    sequences_eval = d_eval.sequence.values[:1]
    labels_eval = d_eval.label.values[:1]

    # sequences_eval = sequences_eval[0]
    # labels_eval = labels_eval[0]

    encoded_eval = tokenizer.batch_encode_plus(
                    sequences_eval,
                    add_special_tokens = True,  # Add '[CLS]' and '[SEP]'
                    padding = 'longest',        # Pad to longest in batch.
                    truncation = True,          # Truncate sentences to `max_length`.
                    max_length = 512,   
                    return_tensors = 'pt',        # Return pytorch tensors.
            )
    encoded_eval = compute_correct_attention_masks(tokenizer_output=encoded_eval)

    print('finished encoding sequences')

    # Take useful info and prepare Dataset
    input_ids_eval = encoded_eval['input_ids']
    attention_masks_eval = encoded_eval['attention_mask']
    labels_eval = torch.tensor(labels_eval)

    dataset_eval = TensorDataset(input_ids_eval, attention_masks_eval, labels_eval) 

    # Prepare Dataloader
    eval_loader = DataLoader(dataset=dataset_eval, sampler=SequentialSampler(dataset_eval), batch_size=batch_size)


    preds = np.zeros([len(dataset_eval),5])
    attention_scores = np.zeros([len(dataset_eval), 12, 512, 512])

    for index, batch in enumerate(eval_loader):
        model.eval()
        batch = tuple(b.to(device) for b in batch)

        with torch.no_grad():
        
            inputs = {'input_ids':      batch[0],
                    'attention_mask': batch[1],
                    'labels':         batch[2],
                    }

            outputs = model(**inputs)
            attention = outputs[-1]
            attention = torch.stack(attention)

    np.save(os.path.join(predict_dir, "atten_scores.npy"), attention)

    exit()

    softmax = torch.nn.Softmax(dim=1)
    probs = softmax(torch.tensor(preds, dtype=torch.float32))[:,1].numpy()

    scores = np.zeros([attention_scores.shape[0], attention_scores.shape[-1]])

    for index, attention_score in enumerate(attention_scores):
        attn_score = []
        for i in range(1, attention_score.shape[-1]-kmer+2):
            attn_score.append(float(attention_score[:,0,i].sum()))

        for i in range(len(attn_score)-1):
            if attn_score[i+1] == 0:
                attn_score[i] = 0
                break

        # attn_score[0] = 0    
        counts = np.zeros([len(attn_score)+kmer-1])
        real_scores = np.zeros([len(attn_score)+kmer-1])
        for i, score in enumerate(attn_score):
            for j in range(kmer):
                counts[i+j] += 1.0
                real_scores[i+j] += score
        real_scores = real_scores / counts
        real_scores = real_scores / np.linalg.norm(real_scores)

        scores[index] = real_scores

    np.save(os.path.join(predict_dir, "atten.npy"), scores)
    np.save(os.path.join(predict_dir, "pred_results.npy"), probs)
    

if __name__ == "__main__":
    main()