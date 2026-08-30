import os
os.environ["MKL_THREADING_LAYER"] = "GNU"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"


from scipy import sparse
import numpy as np
import pandas as pd
import shutil
from sklearn.model_selection import ParameterGrid

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import subprocess
from multiprocessing import Pool

from sklearn.mixture import GaussianMixture
from functools import partial

# from tools import plot_3D, plot_the_PCA_interval, plot_clusters

RANDOM_SEED = 42
COMPETITION_NAME = "ML-2026-Spring-Unsupervised"


def prepare_dir_for_tunnning(algorithm):
    if os.path.isdir(f"{algorithm.__name__}_subs"): shutil.rmtree(f"{algorithm.__name__}_subs")
    os.makedirs(f"{algorithm.__name__}_subs", exist_ok=True)


def pipeline(X, algorithm, model_params):
    param_str = "___".join([f"{k}={v}" for k, v in model_params.items()])
    file_path = f"{algorithm.__name__}_subs/{algorithm.__name__}_{param_str}.csv"

    try:
        model = algorithm(**model_params)
        Y = model.fit_predict(X)
    except Exception:
        print(f"model:{param_str} has uncampatable params")
        return -1
    
    subm = pd.DataFrame({"ID": np.arange(Y.size), "TARGET": Y})
    subm.to_csv(file_path, index=False)

    post_submission(file_path)
    return 0


def post_submission(sub_path):
    command = [
        'kaggle', 'competitions', 'submit',
        '-c', COMPETITION_NAME,
        '-f', sub_path,
        '-m', f'Авто-отправка {sub_path.split("/")[-1]}'
    ]
    
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding='utf-8',
        stdin=subprocess.DEVNULL,
        timeout=20
    )
    
    if result.returncode == 0:
        print(f"Успешно отправлено: {sub_path.split('/')[-1]}\nВывод: {result.stdout}")
    else:
        print(f"Ошибка при отправке {sub_path.split('/')[-1]}\nОшибка: {result.stderr}")


X = sparse.load_npz("../datasets/clusterization/train.npz")
X = X.toarray()
M, N = X.shape
# масштабируем
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# PCA
pca_new = PCA(n_components=40, random_state=RANDOM_SEED)
X_embedded_pca = pca_new.fit_transform(X_scaled)


param_grid = {'n_components': [7, 9, 10, 11, 12], 
              'covariance_type': ['full', 'tied', 'diag'], 
              'init_params': ['kmeans', 'k-means++'],
              'n_init': [3],
              'random_state': [RANDOM_SEED]
}

def main():
    prepare_dir_for_tunnning(GaussianMixture)
    gauss_pipeline = partial(pipeline, X_embedded_pca, GaussianMixture)

    with Pool(processes=7) as pool:
        results = pool.map(gauss_pipeline, list(ParameterGrid(param_grid)))
    print(results)

if __name__ =="__main__":
    main()