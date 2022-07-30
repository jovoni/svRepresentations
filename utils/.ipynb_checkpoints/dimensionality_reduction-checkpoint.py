
from sklearn.decomposition import PCA
import torch

def perform_pca(X, dim=2):
    pca = PCA(n_components=dim)
    pca.fit(X)
    reduced_X = pca.transform(X)
    return reduced_X

if __name__ == "__main__":

    X = torch.load("results/chr21_DEL0.25_DUP0.25_INV0.25_INS0.25_homo.pt")
    reduced_X = perform_pca(X, dim=2)
    print(type(reduced_X))
    print(reduced_X)
