
from transformers import BertConfig, BertForSequenceClassification, DNATokenizer
import pandas as pd
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler

from utils import compute_correct_attention_masks, training_set

def main():
    # INPUT
    csv_name = 'eval'
    batch_size_per_process = 2

    # Info
    model_path = "dnabert6/"

    # Load config, model and tokenizer
    config = BertConfig.from_pretrained(model_path, output_hidden_states=True)
    tokenizer = DNATokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path, config=config)

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
    d_train = pd.read_csv(f"fine_tune_data/train.tsv", sep="\t")
    d_eval = pd.read_csv(f"fine_tune_data/eval.tsv", sep="\t")

    # Prepare tokenized sequences
    sequences_train = d_train.sequence.values
    labels_train = d_train.label.values

    sequences_eval = d_eval.sequence.values
    labels_eval = d_eval.label.values

    # Prepare Dataloader
    training_dataset = torch.utils.data.TensorDataset(sequences_train, labels_train)
    train_loader = DataLoader(dataset=training_dataset, sampler=RandomSampler)

    # Start inference
    with torch.no_grad():
        for batch in tqdm(train_loader, desc="Running inference"):
            model.eval()

            X, y = batch[0], batch[1]

            tokenizer_output = tokenizer.batch_encode_plus(
                    batch,
                    add_special_tokens = True,  # Add '[CLS]' and '[SEP]'
                    padding = 'longest',        # Pad to longest in batch.
                    truncation = True,          # Truncate sentences to `max_length`.
                    max_length = 512,   
                    return_attention_mask = False, # Construct attn. masks.
                    return_tensors = 'pt',        # Return pytorch tensors.
            )

            tokenizer_output = compute_correct_attention_masks(tokenizer_output=tokenizer_output)

            print(tokenizer_output['attention_mask'])

if __name__ == "__main__":
    main()