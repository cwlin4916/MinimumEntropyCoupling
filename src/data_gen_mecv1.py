from functools import partial
import numpy as np
import os
import scipy
from scipy.sparse import issparse
import time
import torch as th
from torch.utils.cpp_extension import load
import torch

# from utils import kl2, entropy2

# mec_cpp = None
# if mec_cpp is None:
#     print("Need to compile first...")
#     mec_cpp = load(name="mec_cpp", sources=[os.path.join(os.path.abspath(os.path.dirname(__file__)), "mec.cpp")],
#                    extra_cflags=['-O3'])
#     print("Finished compiling!")
# entropy = mec_cpp.entropy


def kl2(q, p, prec=18):
    """
    Modified for stability
    """
    res = q * (np.log2(q) - np.log2(p))
    res[q == 0] = 0
    ressum = res.sum()
    return np.around(ressum, decimals=prec)


def entropy2(q, prec=18):
    res = q * np.log2(q)
    res[q == 0] = 0
    ressum = res.sum()
    return -np.around(ressum, decimals=prec)


def greatest_lower_bound(p, q):
    """
  Calculate the greatest lower bound of two distributions p, q using
  cicalese_supermodularity_2002 - Definition 3
  Note: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=992785 FACT 1 is incorrect - see above!
  """
    if p.shape[0] < q.shape[0]:
        p = np.concatenate([p, np.zeros(q.shape[0] - p.shape[0])])
    elif q.shape[0] < p.shape[0]:
        q = np.concatenate([q, np.zeros(p.shape[0] - q.shape[0])])
    p_cumsum = np.cumsum(p, dtype=np.float64)
    q_cumsum = np.cumsum(q, dtype=np.float64)
    z = np.minimum(p_cumsum, q_cumsum, dtype=np.float64)
    z[1:] -= np.minimum(p_cumsum[:-1], q_cumsum[:-1], dtype=np.float64)
    return z

def additive_gap(p, q, M_candidate):
    """
  Here we compare to entropy(glb(p, q)), which is shown to be the lower bound of the entropy of any coupling
  # max(H(p), H(q))  - this is what Kocaoglu compare to, erroneously!
  Find proof for this here: cicalese_how_2017 [Lemma 2]
  """
    # assert not (M_candidate.sum(1)!=p).any(), "incorrect p-marginal!"
    # assert not (M_candidate.sum(0)!=q).any(), "incorrect q-marginal!"
    # lower_bound = max(entropy(p), entropy(q))
    p = -np.sort(-p)
    q = -np.sort(-q)
    lower_bound = entropy2(greatest_lower_bound(p.astype(np.float64), q.astype(np.float64)))
    if scipy.sparse.issparse(M_candidate):
        entropy_M = entropy2(M_candidate.data).astype(np.float64)
    else:
        entropy_M = entropy2(M_candidate.flatten().astype(np.float64))
    return entropy_M - lower_bound


def marginal_error(p, q, M_candidate):
    if scipy.sparse.issparse(M_candidate):
        p_est = M_candidate.tocsr().sum(1).squeeze()
        q_est = M_candidate.tocsc().sum(0).squeeze()
    else:
        p_est = M_candidate.sum(1)
        q_est = M_candidate.sum(0)
    err = max(np.linalg.norm(p - p_est), np.linalg.norm(q - q_est))
    return err


def kl_divergence(p, q):
    return np.sum(np.where(p != 0, p * np.log2(p / q), 0.0))


def jenson_shannon_div(p, q, M_candidate):
    if scipy.sparse.issparse(M_candidate):
        p_est = np.asarray(M_candidate.tocsr().sum(1)).squeeze()
        q_est = np.asarray(M_candidate.tocsc().sum(0)).squeeze()
    else:
        p_est = M_candidate.sum(1)
        q_est = M_candidate.sum(0)
    div_p = (kl_divergence(p_est, p) + kl_divergence(p, p_est)) * 0.5
    div_q = (kl_divergence(q_est, q) + kl_divergence(q, q_est)) * 0.5
    return div_p, div_q


# WORKS, AND IS STABLE! (THIS IS OUR NEW OPTION!)
def mec_kocaoglu_np(p: np.array, q: np.array):
    """
    Algorithm 1:  https://arxiv.org/pdf/1611.04035.pdf
    We adjust Algorithm 1 and follow the advice in the text in order to reconstruct the matrix.

    Supposedly has 1-bit guarantee - unfortunately not clear if equal to kacaoglu2

    We require len(p) == q.
    """
    p = p.copy().astype(np.longdouble)
    p /= p.sum()
    q = q.copy().astype(np.longdouble)
    q /= q.sum()
    assert len(p) == len(q), "len(p) must be equal to len(q)!"
    J = np.zeros((q.shape[0], p.shape[0]), dtype=np.longdouble)  # Joint distribution

    # e = []
    M = np.stack((p, q), 0)
    r = M.max(axis=1).min()
    while r > 0:
        # e.append(r)
        a_i = M.argmax(axis=1)
        M[0, a_i[0]] -= r
        M[1, a_i[1]] -= r
        J[a_i[0], a_i[1]] = r
        r = M.max(axis=1).min()
    return J


def minimum_entropy_coupling(p: np.ndarray, q: np.ndarray, method="kocaoglu", select_col="all", select_row="all",
                             verbose=False, **kwargs):
    global mec_cpp
    assert p.ndim == 1 and q.ndim == 1, "ERROR: batch mode not yet supported!"
    stats = {}
    ret_dct = {}

    # assert p, q are distributions
    if not (np.isclose(p.sum(), np.longdouble(1.0)) and
            np.isclose(np.sum(p[p < 0.0]), np.longdouble(0.0)) and
            np.isclose(q.sum(), np.longdouble(1.0)) and
            np.isclose(np.sum(q[q < 0.0]), np.longdouble(0.0))):
        assert False, "Either q, p (or both) are not proper probability distributions! p: {} q: {}".format(p, q)

    t1 = time.time()
    if p.shape[0] > q.shape[0]:
        q = np.concatenate([q, np.zeros(p.shape[0]-q.shape[0], dtype=np.longdouble)])
    elif q.shape[0] > p.shape[0]:
        p = np.concatenate([p, np.zeros(q.shape[0]-p.shape[0], dtype=np.longdouble)])

    p = p.astype(np.longdouble)
    p /= p.sum()
    q = q.astype(np.longdouble)
    q /= q.sum()

    # Note: We here use a numerically stabilised variant of MEC by Koacoglu et al, 2017
    # Runtime complexity is O(max(p_dim, q_dim)**2)
    M = mec_kocaoglu_np(p, q)

    p_est = M.sum(1)
    q_est = M.sum(0)
    p_est = p_est / p_est.sum()
    q_est = q_est / q_est.sum()

    ret_dct["M_entropy"] = entropy2(M.flatten())
    ret_dct["q_entropy"] = entropy2(q)
    ret_dct["kl_q"] = kl2(q_est, q)
    ret_dct["kl_p"] = kl2(p_est, p)
    ret_dct["p_est"] = p_est
    ret_dct["q_est"] = q_est
    ret_dct["stats"] = stats
    ret_dct["additive_gap"] = ret_dct["M_entropy"] - entropy2(greatest_lower_bound(p, q))
    if select_row not in [None, "all"]:
        ret_dct["M_selected_row"] = M[select_row]
    if select_col not in [None, "all"]:
        ret_dct["M_selected_col"] = M[:, select_col]
    ret_dct["M_colfirst"] = np.transpose(M) ## why transpose? we turn it back to the original form 
    ret_dct["M_rowfirst"] = M
    ret_dct["mec_time[s]"] = time.time() - t1

    return ret_dct


# now we would like to generate a collection of data for training the model. These are triples (p,q,M) where p and q are marginals and M is the joint distribution.
# we will have a uniform function which generates a random distribution of size n 
#fix a seed  
def uniform(n):
    """Generate a uniform distribution of dimension n."""
    cuts = np.sort(np.random.uniform(0, 1, n-1))
    print(cuts) # this is the cuts
    return np.diff(np.concatenate(([0], cuts, [1]))) # diff computes the difference between consecutive elements in the array 
# test the code 
# print(uniform(5))
# # to check this is indeed a distribution we can check that the sum is 1
# print(uniform(5).sum())

def generate(k,m):
    """generate and save dataset of minimum entropy couplings."""
    directory = "data/mec_generation"
    os.makedirs(directory, exist_ok=True)
    num_samples = int(10**k)
    filename = os.path.join(directory, f"mec_data_{m}_10^{k}.pth")
    
    data =[] 
    for _ in range(num_samples):
        p = uniform(m)
        q = uniform(m)
        M = mec_kocaoglu_np(p, q)
        data.append((p,q,M))
    torch.save(data, filename)
    print(f"Data saved to {filename}")
    print("A few elements from the saved data:")

    for i, (p, q, M) in enumerate(data[:min(3, len(data))]):
        print(f"Sample {i}:")
        print("p:", p)
        print("q:", q)
        print("M:", M)


# sample    usage 
if __name__ == "__main__":
    # generate for m from 4 to 10 for k=3 
    for m in range(3, 11):
        generate(5, m)











# # now let us test each of the above functions 
# # we will begin with testing kl2 this computes the kl divergence between two distributions
# # we first create two distributions p and q 
# # we will then call the function kl2 with these two distributions 

# # Test kl2 function
# p = np.array([0.1, 0.9])
# q = np.array([0.2, 0.8])
# print("Our distributions are p:", p, "and q:", q)
# print("KL Divergence (KL2) between p and q:", kl2(q, p))


# # test greatest_lower_bound function this calculates the greatest lower bound of two distributions p, q 
# # Test greatest_lower_bound function
# # this would get more interesting if p, q are of size 3 and are not comparable 
# p = np.array([0.4, 0.3, 0.3])
# q = np.array([0.5, 0.1, 0.4])
# print("Our distributions are p:", p, "and q:", q)
# print("Greatest lower bound of p and q:", greatest_lower_bound(p, q))
 
#  # test additive_gap function this calculates the additive gap of two distributions p, q 
# p=np.array([0.1, 0.9,0])
# q=np.array([0.2, 0.7, 0.1])
# M_candidate = np.array([[0.1, 0.0], [0.0, 0.9]]) # this is the candidate matrix 
# print("Our distributions are p:", p, "and q:", q)
# print("M_candidate:", M_candidate)
# print("Additive gap between p, q and M_candidate:", additive_gap(p, q, M_candidate))  

# now we test mec_kocaoglu_np function this is the minimum entropy coupling algorithm
# MK = mec_kocaoglu_np(p, q)
# print("Minimum entropy coupling matrix of the first version:", MK)
# # also test the second version 
# MK2 = minimum_entropy_coupling(p, q)["M_rowfirst"]
# print("Minimum entropy coupling matrix of the second version:", MK2)
# we can see that the two versions are the same 
# let us extract matrix from the second version 
