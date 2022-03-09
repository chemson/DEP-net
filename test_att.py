import torch
import math
import torch.nn as nn
from tqdm.notebook import tqdm
import numpy as np
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn import (manifold, datasets, decomposition, ensemble, discriminant_analysis, random_projection)
# def test_MultiHeadSelfAttention():
#     model = MultiHeadSelfAttention(dim=64)
#     x = torch.rand(10, 10, 64)  # [batch, tokens, dim]
#     mask = torch.zeros(10, 10)  # tokens X tokens
#     mask[5:8, 5:8] = 1
#     y = model(x, mask)
#     # y = torch.squeeze(y).t()
#     # x = torch.squeeze(x)
#     # pro = torch.matmul(x, y)
#     x = torch.mean(x, dim=0)
#     assert y.shape == x.shape
#     # multihead_attn = nn.MultiheadAttention(60, 4)
#     # x = torch.rand(1, 10, 60)
#     # out_put, weight = multihead_attn(x, x, x)
#
#
#     print("MultiHeadSelfAttentionAISummer OK")


# test_MultiHeadSelfAttention()
class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''
    def __init__(self, temperature, attn_dropout=0.0):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        attn = torch.sum(attn, dim=-1)
        attn = F.softmax(attn, dim=-1)
        return attn

# # Attention
# query = torch.rand(128, 32, 1, 256)
# key = value = torch.rand(128, 16, 1, 256)
# query, key, value = query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)
# multihead_attn = ScaledDotProductAttention(temperature=query.size(2))
# attn_output, attn_weights = multihead_attn(query, key, value)
# attn_output = attn_output.transpose(1, 2)
# print(f'attn_output: {attn_output.size()}, attn_weights: {attn_weights.size()}')

def plot_embedding(X, y, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure()
    # ax = plt.subplot(111)
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], str(y[i]), color=plt.cm.Set1(y[i] / 10.), fontdict={'weight': 'bold', 'size': 9})
    plt.text(X[600, 0], X[600, 1], str(y[600]), color='blue', fontdict={'weight': 'bold', 'size': 9})
    shown_images = np.array([[1., 1.]])  # just something big
    for i in range(X.shape[0]):
        dist = np.sum((X[i] - shown_images) ** 2, 1)
        # if np.min(dist) < 4e-3:
        #     # don't show points that are too close
        #     continue
        # if np.max(dist) > 4e-3:
        #     # don't show points that are too far
        #     continue
        shown_images = np.r_[shown_images, [X[i]]]
    plt.xticks([]), plt.yticks([])
    plt.title(title)
    plt.show()



def tSNE(X, y, title):
    print("Computing t-SNE embedding")
    # tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    tsne = manifold.TSNE()
    X_tsne = tsne.fit_transform(X)
    plot_embedding(X_tsne, y, "tSNE" + title)



# Self-attention
# query = torch.randn(1, 600, 1, 640)
#
# aaa = query
# query = query.transpose(1, 2)
# multihead_attn = ScaledDotProductAttention(temperature=query.size(2))
# attn_output = multihead_attn(query, query, query).squeeze(dim=0)
# query = query.squeeze()
# ans = torch.matmul(attn_output, query)
# ans2 = aaa.squeeze(dim=0).mean(dim=0)
# # attn_output = attn_output.transpose(1, 2)
# test = torch.cat((aaa.squeeze(), ans2),dim=0)
# test = torch.cat((test,ans),dim=0)
# label = [0.0 for i in range(602)]
# label[600] = 1.0
# label[601] = 2.0
# tSNE(test, label, 'x')

