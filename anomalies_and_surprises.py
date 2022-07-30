
import time
import sklearn.mixture
import sklearn.svm
from tqdm import tqdm
import os
from transformers import BertForSequenceClassification, DNATokenizer
import torch
import numpy as np
from torch.utils.data import SequentialSampler, DataLoader

from utils.create_synthetic_dataset import create_homogenous_dataset, create_tsv_dataset
from utils.utils import compute_correct_attention_masks

class SentEncoder:
  def __init__(self, batch_size, model_path='dnabert6/'):

    if torch.cuda.is_available():
        self.device = torch.device("cuda")
    else:
        self.device = torch.device("cpu")
    self.model_name = model_path
    self.batch_size = batch_size
    self.tokenizer = DNATokenizer.from_pretrained(model_path)
    self.model = BertForSequenceClassification.from_pretrained(model_path, output_hidden_states=True).to(self.device)
    self.pad_id = self.tokenizer.pad_token_id

  def contextual_token_vecs(self, sents):
    """Returns: (all_tokens, sentence_token_vecs) where:
    sentence_token_vecs is List[np.array(sentence length, 13, 768)], one array for each sentence.
    Ignore special tokens like [CLS] and [PAD].
    """
    sentence_token_vecs = []

    seq_sampler = SequentialSampler(sents)
    seq_loader = DataLoader(dataset=sents, sampler=seq_sampler, batch_size=self.batch_size)

    for batch in seq_loader:
        with torch.no_grad(): 
            self.model.eval()

            tokenizer_output = self.tokenizer.batch_encode_plus(
                        batch,
                        add_special_tokens = True,  # Add '[CLS]' and '[SEP]'
                        padding = 'longest',        # Pad to longest in batch.
                        truncation = True,          # Truncate sentences to `max_length`.
                        max_length = 512,   
                        return_attention_mask = True, # Construct attn. masks.
                        return_tensors = 'pt',        # Return pytorch tensors.
                )

            model_input = compute_correct_attention_masks(tokenizer_output=tokenizer_output)

            # (num_layers, batch_size, sent_length, 768)
            model_output = self.model(**model_input)
            hidden_states = model_output[1]

            # Create tensor of embeddings
            token_embeddings = torch.stack(hidden_states, dim=0)
            token_embeddings = token_embeddings.permute(1,2,0,3)

        for e in token_embeddings:
            sentence_token_vecs.append(e)

    return sentence_token_vecs

class AnomalyModel:
    def __init__(self, train_sentences, batch_size, model_path='dnabert6/',
      model_type='gmm', n_components=1, covariance_type='full',
      svm_kernel='rbf'):

        self.enc = SentEncoder(model_path=model_path, batch_size=batch_size)
        self.gmms = []

        # Assumes base models have 12+1 layers, large models have 24+1
        self.num_model_layers = 25 if 'large' in model_path else 13

        all_vecs = self.enc.contextual_token_vecs(train_sentences)
        print("Fitting")
        for layer in tqdm(range(self.num_model_layers)):
            sent_vecs = np.vstack([vs[:,layer,:] for vs in all_vecs])

            if model_type == 'gmm':
                gmm = sklearn.mixture.GaussianMixture(n_components=n_components, covariance_type=covariance_type)
            elif model_type == 'svm':
                gmm = sklearn.svm.OneClassSVM(kernel=svm_kernel)

            gmm.fit(sent_vecs)
            self.gmms.append(gmm)


    def gmm_score(self, sentences):
        """Returns (all_tokens, all_scores), where
        all_tokens is List[List[token]]
        all_scores is List[np.array(num layers, |S|)]
        """

        all_vecs = self.enc.contextual_token_vecs(sentences)
        all_scores = []

        for sent_ix in range(len(sentences)):
            # tokens = all_tokens[sent_ix]
            vecs = all_vecs[sent_ix]
            # assert len(tokens) == vecs.shape[0]
            
            layer_scores = []
            for layer in range(self.num_model_layers):
                scores = self.gmms[layer].score_samples(vecs[:, layer, :])
                layer_scores.append(scores)

            all_scores.append(np.array(layer_scores))

        return all_scores


    def eval_sent_pairs(self, sentpairs, layer):
        """Evaluate sentence pairs, assuming first pair is correct one.
        Return list of score differences, positive if the correct one has higher likelihood.
        """
        results = []
        if layer is None:
            for layer in tqdm(range(self.num_model_layers)):
                correct_scores = self.gmm_score([sp[0] for sp in sentpairs])
                correct_scores = [np.sum(sent_scores[layer]) for sent_scores in correct_scores]
                incorrect_scores = self.gmm_score([sp[1] for sp in sentpairs])
                incorrect_scores = [np.sum(sent_scores[layer]) for sent_scores in incorrect_scores]
                results.append([x - y for (x,y) in zip(correct_scores, incorrect_scores)])

            return results
        else:
            correct_scores = self.gmm_score([sp[0] for sp in sentpairs])
            correct_scores = [np.sum(sent_scores[layer]) for sent_scores in correct_scores]
            incorrect_scores = self.gmm_score([sp[1] for sp in sentpairs])
            incorrect_scores = [np.sum(sent_scores[layer]) for sent_scores in incorrect_scores]
            results.append([x - y for (x,y) in zip(correct_scores, incorrect_scores)])

            return results

def main():
    n = 100
    train_ratio = 0.5
    ratios = [0, 1, 1, 1, 1] # all Svs equally, 0 for no Sv applied
    chr_name = "chr21"
    path_folder = f"dataset/surprise"
    same_sequence = False
    original_sequence_first = False
    batch_size = 1
    clean_all = True

    if clean_all:
        filelist = [ f for f in os.listdir(path_folder)]
        for f in filelist:
            os.remove(os.path.join(path_folder, f))
        print("cleaned folder")

    df = create_homogenous_dataset(N=n, ratios=ratios, chromosome_name=chr_name, same_sequence=same_sequence, put_original_sequence_first=original_sequence_first)
    df.to_csv("dataset/surprise/all_data.csv", sep=";", index=False)
    print("created data")

    # Convert to tsv data
    original_seq_df, _ = create_tsv_dataset(df=df, dataset_path=None, ratio_train=1, use_original_seq=True)
    modified_seq_df, _ = create_tsv_dataset(df=df, dataset_path=None, ratio_train=1, use_original_seq=False)
    original_seq_df.to_csv("dataset/surprise/train.tsv", sep="\t", index=False)
    modified_seq_df.to_csv("dataset/surprise/val.tsv", sep="\t", index=False)


    original_sequences = original_seq_df.sequence.values
    modified_sequences = modified_seq_df.sequence.values

    n_train = int(len(original_sequences) * train_ratio)
    train_sentences = original_sequences[:n_train]

    eval_original_s = original_sequences[n_train:]
    eval_modified_s = modified_sequences[n_train:]

    model = AnomalyModel(train_sentences=train_sentences, batch_size=batch_size)

    pairs = [(correct_s, false_s) for correct_s, false_s in zip(eval_original_s, eval_modified_s)]
    print(len(pairs))

    scores = model.eval_sent_pairs(pairs)
    for layer, layer_score in enumerate(scores):
        good_results = 0
        for value in layer_score:
                if value > 0:
                    good_results += 1
        print(f"\nLAYER {layer}:")
        print(f"Accuracy: {100 * good_results/len(layer_score):.2f}%")

if __name__ == "__main__":
    main()