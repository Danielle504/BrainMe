import numpy as np
import torch

def train_cvfold(data, reg, numCC, kernelcca, ktype, gausigma, degree,
                 cutoff, selection):
    nT = data[0].shape[0]
    chunklen = 10 if nT > 50 else 1
    nchunks = int(0.2 * nT / chunklen)
    indchunks = list(zip(*[iter(range(nT))] * chunklen))
    np.random.shuffle(indchunks)
    heldinds = [ind for chunk in indchunks[:nchunks]
                for ind in chunk]
    notheldinds = list(set(range(nT)) - set(heldinds))
    comps = kcca([d[notheldinds] for d in data], reg, numCC,
                 kernelcca=kernelcca, ktype=ktype,
                 gausigma=gausigma, degree=degree)
    cancorrs, ws, ccomps = recon([d[notheldinds] for d in data], comps,
                                 kernelcca=kernelcca)
    preds, corrs = predict([d[heldinds] for d in data], ws, cutoff=cutoff)
    fold_corr_mean = []
    for corr in corrs:
        corr_idx = np.argsort(corr)[::-1]
        corr_mean = corr[corr_idx][:selection].mean()
        fold_corr_mean.append(corr_mean)
    return np.mean(fold_corr_mean)

class CCA(_CCABase):
    def __init__(self, reg=0., numCC=10, kernelcca=True, ktype=None,
                 verbose=True, cutoff=1e-15):
        super(CCA, self).__init__(reg=reg, numCC=numCC, kernelcca=kernelcca,
                                  ktype=ktype, verbose=verbose, cutoff=cutoff)

    def train(self, data):
        return super(CCA, self).train(data)


def _make_kernel(d, normalize=True, ktype='linear', gausigma=1.0, degree=2):
    d = np.nan_to_num(d)
    cd = _demean(d)
    if ktype == 'linear':
        kernel = np.dot(cd, cd.T)
    elif ktype == 'gaussian':
        from scipy.spatial.distance import pdist, squareform
        pairwise_dists = squareform(pdist(d, 'euclidean'))
        kernel = np.exp(-pairwise_dists ** 2 / 2 * gausigma ** 2)
    elif ktype == 'poly':
        kernel = np.dot(cd, cd.T) ** degree
    kernel = (kernel + kernel.T) / 2.
    if normalize:
        kernel = kernel / np.linalg.eigvalsh(kernel).max()
    return kernel

if __name__ == "__main__":
  mlp_vector_transform.py
