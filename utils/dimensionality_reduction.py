
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch

def perform_pca(X, dim=2):
    pca = PCA(n_components=dim)
    pca.fit(X)
    reduced_X = pca.transform(X)
    return reduced_X

def perform_tsne(X, dim=2, perplexity=30, early_exaggeration=12, learning_rate=200):
    tsne = TSNE(n_components=dim, perplexity=perplexity, early_exaggeration=early_exaggeration, learning_rate=learning_rate)
    reduced_X = tsne.fit_transform(X)
    return reduced_X

if __name__ == "__main__":

    X = torch.load("results/chr21_DEL0.25_DUP0.25_INV0.25_INS0.25_homo.pt")
    reduced_X = perform_pca(X, dim=2)
    print(type(reduced_X))
    print(reduced_X)
