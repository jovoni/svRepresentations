
import torch

def extract_sentence_embeddings_from_words(word_embeddings, type):

    if type == "avg":
        # Average
        sentence_embedding = torch.mean(word_embeddings, dim=0)
        return sentence_embedding
    else:
        raise Exception("Type not know")

def extract_word_vectors_from_tokens(tokens_embeddings, type):
    n_tokens = tokens_embeddings.shape[0] # how many tokens do we have?
    layer_length = tokens_embeddings.shape[2] # how long is a layer embedding?

    if type == "first_layer":
        # Use only first layer

        length_embedding = 1 * layer_length
        token_vecs = torch.empty([n_tokens, length_embedding])

        for i, token in enumerate(tokens_embeddings):
            token_vecs[i] = token[0]
        return token_vecs   

    elif type == "sum_all":
        # Sum from second to last layer
        
        length_embedding = 1 * layer_length
        token_vecs = torch.empty([n_tokens, length_embedding])

        for i, token in enumerate(tokens_embeddings):
            sum_vec = torch.sum(token, dim=0)
            token_vecs[i] = sum_vec
        return token_vecs      

    elif type == "last_layer":
        # Use only last layer
        
        length_embedding = 1 * layer_length
        token_vecs = torch.empty([n_tokens, length_embedding])

        for i, token in enumerate(tokens_embeddings):
            token_vecs[i] = token[-1]
        return token_vecs 

    elif type == "sum_last_4":
        # Sum last 4 layers
        
        length_embedding = 1 * layer_length
        token_vecs = torch.empty([n_tokens, length_embedding])

        for i, token in enumerate(tokens_embeddings):
            sum_vec = torch.sum(token[-4:], dim=0)
            token_vecs[i] = sum_vec
        return token_vecs   

    elif type == "concat_last_4":
        # Concatenate last 4 hidden layers for every token

        length_embedding = 4 * layer_length # 4 because 4 layers
        token_vecs = torch.empty([n_tokens, length_embedding])

        for i, token in enumerate(tokens_embeddings):
            cat_vec = torch.cat((token[-1], token[-2], token[-3], token[-4]), dim=0)
            # Use `cat_vec` to represent `token`.
            token_vecs[i] = cat_vec
        return token_vecs

    else:
        raise Exception("Type not know")