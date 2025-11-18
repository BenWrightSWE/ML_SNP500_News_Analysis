import pandas as pd
from scipy.sparse import csr_matrix, vstack
from sklearn.model_selection import train_test_split
import re

# changing how python finds the paths - change from jas
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed" / "vector_news.csv"

# converts a string sparse matrix to a sparse matrix

def parse_sparse_matrix_string(s, shape=(1, 22142)):
    pattern = r'\(0, (\d+)\)\s+([\d.e+-]+)'
    matches = re.findall(pattern, s)

    if not matches:
        return csr_matrix(shape)

    cols = [int(m[0]) for m in matches]
    data = [float(m[1]) for m in matches]
    rows = [0] * len(cols)

    return csr_matrix((data, (rows, cols)), shape=shape)

# import the data:

vector_news = pd.read_csv(DATA, header = 0)

# converts string representations to sparse matrices

sparse_matrices = [parse_sparse_matrix_string(s) for s in vector_news["article_vector"]]

# makes array of all the sparse matrices and an array of the djia values

vectorized_articles = vstack(sparse_matrices)
djia_labels = vector_news["djia"].values

# split data

article_train, article_test, djia_train, djia_test = train_test_split(
    vectorized_articles, djia_labels, test_size=0.25, random_state=27)