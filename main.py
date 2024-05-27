import random
import numpy as np
import torch
import scipy.sparse as sp
import os
import argparse
from utils import csr2torch, recall_at_k, ndcg_at_k, normalize_sparse_adjacency_matrix

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
current_directory = os.getcwd()

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset",
    type=str,
    default="gowalla",
    help="Either gowalla, yelp, amazon, or ml-1m",
)
parser.add_argument(
    "--verbose",
    type=int,
    default=1,
    help="Whether to print the results or not. 1 prints the results, 0 does not.",
)
parser.add_argument("--alpha", type=float, default=0.5, help="For normalization of R")
parser.add_argument("--power", type=float, default=1, help="For normalization of P")


random.seed(2022)
np.random.seed(2022)


if __name__ == "__main__":
    args = parser.parse_args()
    if args.verbose:
        print(f"Device: {device}")
    dataset = args.dataset
    path_tr = f"{current_directory}/dataset/{dataset}_train.npz"
    path_ts = f"{current_directory}/dataset/{dataset}_test.npz"
    R_tr = csr2torch(sp.load_npz(path_tr)).to(device)
    R_ts = csr2torch(sp.load_npz(path_ts)).to(device)

    n_users = R_tr.shape[0]
    n_items = R_tr.shape[1]
    if args.verbose:
        print(f"number of users: {n_users}")
        print(f"number of items: {n_items}")

    n_inters = torch.nonzero(R_tr._values()).cpu().size(0) + torch.nonzero(
        R_ts[0]._values()
    ).cpu().size(0)

    if args.verbose:
        print(f"number of overall ratings: {n_inters}")

    R_norm = normalize_sparse_adjacency_matrix(R_tr.to_dense(), args.alpha)
    R = R_tr.to_dense()
    P = R_norm.T @ R_norm
    P.data **= args.power
    P = P.to(device=device).float()
    R = R.to(device=device).float()

    # Our model
    results = R @ (P)

    # Now get the results
    gt_mat = R_ts.to_dense()
    results = results + (-99999) * R_tr.to_dense()
    gt_mat = gt_mat.cpu().detach().numpy()
    results = results.cpu().detach().numpy()

    # print(f"alpha: {a}, p: {p} ")
    print(f"Recall@20: {recall_at_k(gt_mat, results, k=20):.4f}")
    print(f"NDCG@20: {ndcg_at_k(gt_mat, results, k=20):.4f}")
