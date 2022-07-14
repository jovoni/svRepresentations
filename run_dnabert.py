
from embeddings import extract_sentence_embeddings_from_words, extract_word_vectors_from_tokens
from utils import assign_to_device, preprocess_sequence
from transformers import BertConfig, BertForSequenceClassification, DNATokenizer
import pandas as pd
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader, SequentialSampler

MODEL_CLASSES = {
    "dna": (BertConfig, BertForSequenceClassification, DNATokenizer)
}

def main():
    # INPUT
    csv_name = 'eval'
    batch_size_per_process = 16
    embedding_type = 'concat_last_4'

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
    d = pd.read_csv(f"fine_tune_data/{csv_name}.tsv", sep="\t")

    # Prepare tokenized sequences
    sequences = d.sequence.values
    sv_types = d.label.values

    # Prepare Dataloader
    seq_sampler = SequentialSampler(sequences)
    label_sampler = SequentialSampler(sv_types)

    batch_loader = DataLoader(dataset=sequences, sampler=seq_sampler, batch_size=batch_size_per_process)
    label_loader = DataLoader(dataset=sv_types, sampler=label_sampler, batch_size=batch_size_per_process)

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

    torch.save(preds, f"results/{csv_name}_{embedding_type}.pt")
    torch.save(preds, f"results/{csv_name}_{embedding_type}_labels.pt")

if __name__ == "__main__":
    main()