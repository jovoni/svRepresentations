
from utils import assign_to_device, preprocess_sequence
from transformers import BertConfig, BertForSequenceClassification, DNATokenizer
import pandas as pd
from tqdm import tqdm
import torch

MODEL_CLASSES = {
    "dna": (BertConfig, BertForSequenceClassification, DNATokenizer)
}

def main():
    # Info
    config_class, model_class, tokenizer_class = BertConfig, BertForSequenceClassification, DNATokenizer
    tokenizer_name = "dna6"
    model_path = "dnabert6/"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # Load tokenizer
    tokenizer = tokenizer_class.from_pretrained(tokenizer_name)

    # Read sequences
    csv_name = 'chr21_DEL0.25_DUP0.25_INV0.25_INS0.25_etero'
    d = pd.read_csv(f"dataset/{csv_name}.csv", sep=";")
    sequences = d.NEW_SEQ.values

    preprocessed_sequences = [preprocess_sequence(s) for s in sequences]

    batch_to_process = preprocessed_sequences[:12]
    
    output = tokenizer.batch_encode_plus(batch_to_process, return_token_type_ids=True, return_attention_mask=False, return_tensors="pt")
    output = assign_to_device(tokenizer_output=output, device=device)

if __name__ == "__main__":
    main()