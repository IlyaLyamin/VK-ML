from scipy import sparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import os
import time
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
os.environ["MKL_THREADING_LAYER"] = "GNU"


def plot_3D(X, colors=None, title=None, size=2):    
    fig = px.scatter_3d(
        x=X[:, 0], 
        y=X[:, 1], 
        z=X[:, 2] if X.shape[1] >= 3 else np.zeros(X.shape[0]),
        color=colors,
        title=title,
        size=[size for i in range(X.shape[0])],
        hover_data=None,  # убираем дополнительные данные
        hover_name=None
    )  # убираем имя при наведении
    fig.write_html(f"temp_3d_plot_{title}.html")
    fig.show()


def plot_the_PCA_interval(pca_obj, start, stop, step, show_ratios=False, print_cum=False):
    interval = np.arange(start, stop, step)

    explained_variance_ratio = pca_obj.explained_variance_ratio_[interval]
    explained_variance_ratio_reduced = explained_variance_ratio.copy()[1::]
    rations = np.insert(explained_variance_ratio[:-1:]/explained_variance_ratio_reduced, 0, 0)
    cumulative = np.cumsum(pca_obj.explained_variance_ratio_)[interval]
    components = np.arange(0, len(explained_variance_ratio))*step + start

    plt.figure(figsize=(8, 5))
    plt.bar(components, explained_variance_ratio, alpha=0.7, label='Individual')
    plt.plot(components, cumulative, 'ro-', label='Cumulative')
    plt.xlabel('Number of Components')
    plt.ylabel('Explained Variance Ratio')
    plt.xticks(components)
    plt.legend()
    plt.grid(True)
    plt.show()

    if show_ratios or print_cum:
        for comp, cum, ratio in zip(components, cumulative, rations):
            string = f"Компонент {comp}: "
            if show_ratios: string += f"cum={cum:.4f}; "
            if print_cum: string += f"ratio={ratio:.4f}; "
            print(string)


def plot_clusters(data, algorithm, args, kwds):
    start_time = time.time()
    labels = algorithm(*args, **kwds).fit_predict(data)
    subm = pd.DataFrame({"ID": np.arange(labels.size), "TARGET": labels})
    subm.to_csv(f"subm_{algorithm.__name__}.csv", index=False)
    print(labels[:10])
    end_time = time.time()
    palette = sns.color_palette('deep', np.unique(labels).max() + 1)
    colors = [palette[x] if x >= 0 else (0.0, 0.0, 0.0) for x in labels]
    plot_3D(data, colors, title=f"{algorithm.__name__}")
    print('Clusters found by {}'.format(str(algorithm.__name__)))
    print('Clustering took {:.2f} s'.format(end_time - start_time))
