
from utils.create_synthetic_dataset import create_homogenous_dataset, create_tsv_dataset
import os
from utils.embeddings import extract_sentence_embeddings_from_words, extract_word_vectors_from_tokens
from transformers import BertForSequenceClassification, DNATokenizer
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import SequentialSampler, DataLoader
from tqdm import tqdm
import numpy as np
from utils.utils import assign_to_device


def create_data(N, n_sequences, ratios, chr_name, path_folder, put_original_sequence_first):
    for i in range(n_sequences):
        df = create_homogenous_dataset(N=N, ratios=ratios, chromosome_name=chr_name, same_sequence=True, put_original_sequence_first=put_original_sequence_first)
        df_path = f"{path_folder}/all_data_{i}.csv"
        if not os.path.exists(df_path):
            df.to_csv(df_path, sep=";", index=False)
            tsv_dataset, _ = create_tsv_dataset(dataset_path=df_path, ratio_train=1)
            tsv_dataset.to_csv(f"{path_folder}/data_{i}.tsv", sep="\t")

def prepare_data_for_model(path_folder, batch_size, index):
    # Read sequences
    d = pd.read_csv(f"{path_folder}/data_{index}.tsv", sep="\t")

    # Prepare tokenized sequences
    sequences = d.sequence.values
    sv_types = d.label.values

    # Prepare Dataloader
    seq_sampler = SequentialSampler(sequences)
    label_sampler = SequentialSampler(sv_types)

    batch_loader = DataLoader(dataset=sequences, sampler=seq_sampler, batch_size=batch_size)
    label_loader = DataLoader(dataset=sv_types, sampler=label_sampler, batch_size=batch_size)

    return batch_loader, label_loader

def make_inference(path_folder, batch_size, embedding_type, n_sequences):
    # Info
    model_path = "dnabert6/"

    # Load config, model and tokenizer
    tokenizer = DNATokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path, 
                                                        num_labels=5, # types of SV
                                                        output_attentions=False,
                                                        output_hidden_states=True)

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

    for i in range(n_sequences):

        batch_loader, label_loader = prepare_data_for_model(path_folder=path_folder, batch_size=batch_size, index=i)

        # Start inference
        preds = None
        out_label_ids = None
        
        with torch.no_grad():
            for batch, labels in tqdm(zip(batch_loader, label_loader), desc="Running inference"):
                model.eval()

                tokenizer_output = tokenizer.batch_encode_plus(
                        batch,
                        add_special_tokens = True,  # Add '[CLS]' and '[SEP]'
                        padding = 'longest',        # Pad to longest in batch.
                        truncation = True,          # Truncate sentences to `max_length`.
                        max_length = 512,   
                        return_attention_mask = True, # Construct attn. masks.
                        return_tensors = 'pt',        # Return pytorch tensors.
                )

                model_input = assign_to_device(tokenizer_output=tokenizer_output, device=device)

                # Feed model and extract hidden states
                model_output = model(**model_input)
                hidden_states = model_output[1]

                # Create tensor of embeddings
                token_embeddings = torch.stack(hidden_states, dim=0)
                token_embeddings = token_embeddings.permute(1,2,0,3)

                for e in token_embeddings:
                    word_embeddings = extract_word_vectors_from_tokens(e, type=embedding_type)
                    sentence_embedding = extract_sentence_embeddings_from_words(word_embeddings=word_embeddings, type="avg")

                    if preds is None:
                        preds = sentence_embedding.detach().cpu()
                        out_label_ids = labels.detach().cpu()
                    else:
                        preds = np.append(preds, sentence_embedding.detach().cpu().numpy(), axis=0)
                        out_label_ids = np.append(out_label_ids, labels.detach().cpu().numpy(), axis=0)

        torch.save(preds, f"results/same_sequence/embeddings_{embedding_type}_{i}.pt")
        torch.save(preds, f"results/same_sequence/embeddings_{embedding_type}_labels_{i}.pt")


def main():
    n = 200
    ratios = [0, 1, 1, 1, 1] # all Svs equally, 0 for no Sv applied
    chr_name = "chr21"
    n_sequences = 2
    path_folder = f"dataset/same_sequence"
    batch_size = 1
    emb_types = ["first_layer", "concat_last_4", "sum_all", "last_layer", "sum_last_4"]
    original_sequence_first = True
    clean_all = True

    if clean_all:
        filelist = [ f for f in os.listdir(path_folder)]
        for f in filelist:
            os.remove(os.path.join(path_folder, f))

    create_data(N=n, ratios=ratios, chr_name=chr_name, path_folder=path_folder, n_sequences=n_sequences, put_original_sequence_first=original_sequence_first)
    for emb_type in emb_types:
        make_inference(path_folder=path_folder, batch_size=batch_size, embedding_type=emb_type, n_sequences=n_sequences)

if __name__ == "__main__":
    main()