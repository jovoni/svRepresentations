
from embeddings import extract_sentence_embeddings_from_words, extract_word_vectors_from_tokens
from utils import prepare_bert_input
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

    # Load tokenizer
    tokenizer = tokenizer_class.from_pretrained(tokenizer_name)

    # Read sequences
    csv_name = 'chr21_DEL0.25_DUP0.25_INV0.25_INS0.25_etero'
    d = pd.read_csv(f"dataset/{csv_name}.csv", sep=";")
    sequences = d.NEW_SEQ.values  

    # Load model
    model = model_class.from_pretrained(model_path, output_hidden_states=True)
    model.to(device)
    model.eval()

    types_embeddings = ['sum_last_4', 'concat_last_4', "sum_all"]
    embedding_factor = [1, 4, 1]

    n_sentences = len(sequences)
    length_embeddings = [768 * f for f in embedding_factor] # comes from bert architecture

    # Make inference
    with torch.no_grad():

        for embedd_type, l_embedding in tqdm(zip(['sum_last_4', 'concat_last_4', "sum_all"], length_embeddings)):
            
            embeddings_of_sentences = torch.empty([n_sentences, l_embedding])

            for i, seq in tqdm(enumerate(sequences)):
                inputs = prepare_bert_input(seq=seq, tokenizer=tokenizer).to(device)		
		outputs = model(**inputs)
                hidden_states = outputs[1]

                # Concatenate the tensors for all layers. We use `stack` here to
                # create a new dimension in the tensor.
                token_embeddings = torch.stack(hidden_states, dim=0)
                token_embeddings = torch.squeeze(token_embeddings, dim=1)
                # Swap dimensions 0 and 1.
                token_embeddings = token_embeddings.permute(1,0,2)

                word_embeddings = extract_word_vectors_from_tokens(token_embeddings, type=embedd_type)
                sentence_embedding = extract_sentence_embeddings_from_words(word_embeddings=word_embeddings, type="avg")

                embeddings_of_sentences[i] = sentence_embedding

            torch.save(embeddings_of_sentences, f"results/{csv_name}_{embedd_type}.pt")

if __name__ == "__main__":
    main()
