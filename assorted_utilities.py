
import re   #regular expression string searching
import pandas as pd #dataframe, data management
import pydicom  #DICOM reader
import numpy as np #general purpose vectorized math library
import seaborn as sns


from sklearn.metrics import recall_score, f1_score, precision_score
from sklearn.metrics.scorer import make_scorer

# weighted recall score
#
def custom_scorer(y_true, y_pred,weight=1):
    score = np.nan

    try:
        score = f1_score(y_true,y_pred,average="weighted")
        recall_score_in = recall_score(y_true,y_pred,average="binary",labels=[0,1],pos_label=0)
        score += weight*recall_score_in

    except ValueError: 
        print("Error")

    return score / (weight+1)

#functions for plotting 2d decision boundary
#
def _make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy

def _plot_contours(ax, model, xx, yy, **params):
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

def make_2d_decision_boundary(model,X,y,feature_labels):
    fig, ax = plt.subplots(figsize=[4,4])

    # Set-up grid for plotting.
    X0, X1 = X[:, 0], X[:, 1]
    xx, yy = _make_meshgrid(X0, X1)

    _plot_contours(ax, model, xx, yy, cmap="RdYlBu", alpha=0.7)
    ax.scatter(X0, X1, c=y, cmap='autumn', s=20, edgecolors='k')
    ax.set_xlabel(feature_labels[0],fontsize=18)
    ax.set_ylabel(feature_labels[1],fontsize=18)
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(f"{feature_labels[0]} vs {feature_labels[1]})
    ax.legend()
    plt.show()
    
## score collection 
#

