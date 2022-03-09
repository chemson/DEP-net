import collections
import pickle
import random
import numpy as np
import matplotlib.pyplot as plt

from sklearn import preprocessing
import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import math
import torch.nn.functional as F
import torch.optim as optim
from numpy import linalg as LA
from tqdm.notebook import tqdm
import os
from sklearn.svm import SVC
from deltaencoder import Encoder,Decoder
from sklearn.linear_model import LogisticRegression
from torch.utils.data import TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.linalg import norm
from torch.utils.data import DataLoader
from test_att import ScaledDotProductAttention
# ========================================
#      loading datas
cos = nn.CosineSimilarity(dim=1, eps=1e-6)

def centerDatas(datas):
    datas[:, :n_lsamples] = datas[:, :n_lsamples, :] - datas[:, :n_lsamples].mean(1, keepdim=True)
    datas[:, :n_lsamples] = datas[:, :n_lsamples, :] / torch.norm(datas[:, :n_lsamples, :], 2, 2)[:, :, None]
    datas[:, n_lsamples:] = datas[:, n_lsamples:, :] - datas[:, n_lsamples:].mean(1, keepdim=True)
    datas[:, n_lsamples:] = datas[:, n_lsamples:, :] / torch.norm(datas[:, n_lsamples:, :], 2, 2)[:, :, None]

    return datas

def scaleEachUnitaryDatas(datas):

    norms = torch.linalg.norm(datas, ord=1, dim=2, keepdim=True)
    return datas/norms


def QRreduction(datas):

    ndatas = torch.qr(datas.permute(0,2,1)).R
    ndatas = ndatas.permute(0,2,1)
    return ndatas


class Model:
    def __init__(self, n_ways):
        self.n_ways = n_ways

# ---------  GaussianModel
class GaussianModel(Model):
    def __init__(self, n_ways, lam):
        super(GaussianModel, self).__init__(n_ways)
        self.mus = None         # shape [n_runs][n_ways][n_nfeat]
        self.lam = lam

    def clone(self):
        other = GaussianModel(self.n_ways)
        other.mus = self.mus.clone()
        return self

    def cuda(self):
        self.mus = self.mus.cuda()

    def initFromLabelledDatas(self):
        self.mus = ndatas.reshape(n_runs, n_shot+n_queries, n_ways, n_nfeat)[:,:n_shot,].mean(1)
        #print(self.mus.shape) (10000,5,100)

    def updateFromEstimate(self, estimate, alpha):

        Dmus = estimate - self.mus
        self.mus = self.mus + alpha * (Dmus)

    def compute_optimal_transport(self, M, r, c, epsilon=1e-6):

        r = r.cuda()#（10000,75）
        c = c.cuda()#（10000,5）
        n_runs, n, m = M.shape#10000,75,5
        P = torch.exp(- self.lam * M)#（10000,75,5）
        P /= P.view((n_runs, -1)).sum(1).unsqueeze(1).unsqueeze(1)#同上

        u = torch.zeros(n_runs, n).cuda()
        maxiters = 1000
        iters = 1
        # normalize this matrix
        while torch.max(torch.abs(u - P.sum(2))) > epsilon:
            u = P.sum(2)#（10000,75）
            P *= (r / u).view((n_runs, -1, 1))#（10000,75,5）
            P *= (c / P.sum(1)).view((n_runs, 1, -1))
            if iters == maxiters:
                break
            iters = iters + 1
        return P, torch.sum(P * M)

    def getProbas(self):
        #print(self.mus.shape)(10000,5,100)
        # compute squared dist to centroids [n_runs][n_samples][n_ways]
        torch.cuda.empty_cache()
        dist = torch.zeros(n_runs, n_samples, n_ways).cuda()
        for i in range(n_runs):
            dist[i] = (ndatas[i].unsqueeze(1)-self.mus[i].unsqueeze(0)).norm(dim=2).pow(2)

        p_xj = torch.zeros_like(dist)
        r = torch.ones(n_runs, n_usamples)#（10000,75）
        c = torch.ones(n_runs, n_ways) * n_queries#（10000,5）

        p_xj_test, _ = self.compute_optimal_transport(dist[:, n_lsamples:], r, c, epsilon=1e-6)#（10000,75,5）
        p_xj[:, n_lsamples:] = p_xj_test#（10000,100,5）

        p_xj[:,:n_lsamples].fill_(0)
        p_xj[:,:n_lsamples].scatter_(2,labels[:,:n_lsamples].unsqueeze(2), 1)#对对应的维度将相应的索引的值改成1

        return p_xj

    def estimateFromMask(self, mask):

        emus = mask.permute(0,2,1).matmul(ndatas).div(mask.sum(dim=1).unsqueeze(2))#matmul返回矩阵的乘积，div除法

        return emus


# =========================================
#    MAP
# =========================================

class MAP:
    def __init__(self, alpha=None):

        self.verbose = True
        self.progressBar = False
        self.alpha = alpha

    def getAccuracy(self, probas):
        olabels = probas.argmax(dim=2)#(10000,100)
        matches = labels.eq(olabels).float()#(10000,100)值为1或0
        acc_test = matches[:, n_lsamples:].mean(1)  #（10000）

        m = acc_test.mean().item()
        pm = acc_test.std().item() * 1.96 / math.sqrt(n_runs)
        return m, pm

    def performEpoch(self, model, epochInfo=None):

        p_xj = model.getProbas()#（10000,100,5）
        self.probas = p_xj#M*

        # if self.verbose:
        #     print("accuracy from filtered probas", self.getAccuracy(self.probas))

        m_estimates = model.estimateFromMask(self.probas)#μ

        # update centroids
        model.updateFromEstimate(m_estimates, self.alpha)

        if self.verbose:
            op_xj = model.getProbas()
            acc = self.getAccuracy(op_xj)
            # print("output model accuracy", acc)

    def loop(self, model, n_epochs=20):
        self.probas = model.getProbas()
        # if self.verbose:
        #     print("initialisation model accuracy", self.getAccuracy(self.probas))

        if self.progressBar:
            if type(self.progressBar) == bool:
                pb = tqdm(total = n_epochs)
            else:
                pb = self.progressBar

        for epoch in range(1, n_epochs+1):
            # if self.verbose:
            #     print("----- epoch[{:3d}] lr_p: {:0.3f}".format(epoch, self.alpha))
            self.performEpoch(model, epochInfo=(epoch, n_epochs))
            if (self.progressBar): pb.update()

        # # get final accuracy and return it
        # op_xj = model.getProbas()#(10000,100,5)
        # acc = self.getAccuracy(op_xj)
        return model.mus



class FinalModel(nn.Module):
    def __init__(self, in_shape, out_shape):
        super(FinalModel, self).__init__()
        self.classifier = nn.Sequential(
              nn.Linear(in_shape, 1024),
              nn.Tanh(),
              nn.Dropout(),
              nn.Linear(1024, 512),
              nn.BatchNorm1d(512),
              nn.Tanh(),
              nn.Dropout(),
              nn.Linear(512, out_shape),
              # nn.Softmax()
            )
    def forward(self, x):
      x = self.classifier(x)
      return x






def optimizer(encoder, decoder, lr, decay):
    optimizer = torch.optim.Adam([{'params': encoder.parameters()},
                                      {'params': decoder.parameters()}], lr=lr,
                                 weight_decay=decay)
    return optimizer

def loss(features_batch, proto_classes):
        loss = nn.L1Loss().cuda()  # 取预测值和真实值的绝对误差的平均数

        pred_noise = encoder(features_batch, proto_classes)
        pred_x = decoder(proto_classes, pred_noise)

        # assert self.pred_noise.shape == self.pred_x.shape
        abs_diff = loss(features_batch, pred_x)

        w = torch.pow(abs_diff, 2)  # 实现张量和标量之间逐元素求指数操作，这里指平方
        w = w / torch.norm(w)  # 求2范数

        loss = w * abs_diff

        return loss, pred_x


def euclidean_metric(a, b):
    n = a.shape[0]  # a(5,640)
    m = b.shape[0]  # b(80,640)
    a = a.unsqueeze(1).expand(n, m, -1)  # a(5,80,640)
    b = b.unsqueeze(0).expand(n, m, -1)  # b(5,80,640)
    # print((-((a - b)**2)).shape) torch.Size([5, 80, 640])
    logits = ((a - b) ** 2).sum(dim=2)  # Tensor(5,80)将第二维度累加
    # cosin = nn.CosineSimilarity(dim=2,eps=1e-6)
    # logits = cosin(a, b)
    return logits


def next_batch(data, label, start, end):
    if start == 0:
        for l in range(label.shape[1]):
            inds = np.where(label[:, l])[0]  # 输出满足条件 (即非0) 元素的坐标
            inds_pairs = np.random.permutation(inds)
            data[inds, :] = data[inds_pairs, :]
    if end >= data.shape[0]:
        end = data.shape[0]
    return data[start:end]




import logging
import time
import os
def log_creater(output_dir):
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)
  log_name = '{}.log'.format(time.strftime('%Y-%m-%d-%H-%M'))
  final_log_file = os.path.join(output_dir,log_name)
  # creat a log
  log = logging.getLogger('train_log')
  log.setLevel(logging.DEBUG)

  # FileHandler
  file = logging.FileHandler(final_log_file)
  file.setLevel(logging.DEBUG)

  # StreamHandler
  stream = logging.StreamHandler()
  stream.setLevel(logging.DEBUG)

  # Formatter
  formatter = logging.Formatter(
    '[%(asctime)s][line: %(lineno)d] ==> %(message)s')

  # setFormatter
  file.setFormatter(formatter)
  stream.setFormatter(formatter)

  # addHandler
  log.addHandler(file)
  log.addHandler(stream)

  log.info('creating {}'.format(final_log_file))
  return log


def save_model(encoder, decoder, save_dir):
    if not os.path.exists(os.path.dirname(save_dir)):
        os.mkdir(os.path.dirname(save_dir))
    torch.save({
        'encoder_state_dict': encoder.state_dict(),
        'decoder_state_dict': decoder.state_dict(),
    }, save_dir)
# def val():
#         lam = 10
#         model = GaussianModel(n_ways, lam)
#         model.initFromLabelledDatas()
#
#         alpha = 0.2
#         optim = MAP(alpha)
#
#         optim.verbose = True
#         optim.progressBar = True
#
#         pro = optim.loop(model, n_epochs=20)#(100,5,640)
#         encoder.eval()
#         decoder.eval()
#         Accuracy = 0.0
#         for runs in range(12, 24):
#             logits = euclidean_metric(pro[runs], train_pro)
#             new_feature_train = torch.arange(384000.0).reshape(600, 640).cuda()
#             for j in range(5):
#                 class_id = logits[j].argmin().item()
#                 # print(ndatas.shape)#(100,3000,640),5*600=3000,100runs
#                 # print(feature_train.shape)#(48000,640)
#                 shot_features = ndatas[runs][:n_shot*n_ways]
#                 class_train = feature_train[class_id*600:class_id*600+600]
#                 proty = class_train.mean(dim=0).repeat(600, 1)
#                 noise = encoder(class_train, proty)
#                 new_feature = decoder(proty, noise)
#                 new_feature_train = torch.cat((new_feature_train, new_feature), dim=0)
#             new_feature_train = torch.cat((new_feature_train[600:], shot_features), dim=0)
#             label = [0 for i in range(3025)]
#             j = -1.0
#             for i in range(3000):
#                 if i % 605 == 0:
#                     j = j + 1.0
#                 label[i] = j
#             for i in range(3000, 3025):
#                 label[i] = i % 5
#
#             train_set = TensorDataset(new_feature_train, torch.tensor(label).cuda())
#             train_loader_cls = torch.utils.data.DataLoader(train_set, batch_size=50, shuffle=True, num_workers=0)
#             netF = FinalModel(640, 5).cuda()
#             criterion = nn.CrossEntropyLoss()
#             optimizer_F = torch.optim.Adam(netF.parameters())
#
#
#             for epoch in range(20):
#                 counter = 0
#                 loss_S = 0
#                 for (data_x, data_y) in train_loader_cls:
#                     netF.train()
#                     data_x = data_x.cuda()
#                     data_y = data_y.long().cuda()
#                     y_pred = netF(data_x).cuda()
#                     loss_cls = criterion(y_pred, data_y)
#                     optimizer_F.zero_grad()
#                     loss_cls.backward(retain_graph=True)
#                     optimizer_F.step()
#                     loss_S += loss_cls.item()
#                     counter += 1
#
#                 # if epoch % 2 == 0:
#                 #     print("Epoch: ", epoch, "Loss: ", loss_S / counter)
#             netF.eval()
#             loss_S = loss_S / counter
#             netF.eval()
#             loss_S = loss_S / counter
#             y_pred = netF(ndatas[runs][25:])
#             y_pred = torch.max(y_pred.data, 1).indices#(3000)
#             query_label = labels[runs][25:]
#             acc_Q = (y_pred == query_label).sum().item() / query_label.size(0)
#             Accuracy += acc_Q
#             del netF
#             print("Total Loss on Training", loss_S, "Accuracy on Query set", Accuracy/(runs+1))
#         print("--------------------Total Accuracy----------", " accuracy: ", Accuracy / 20)

# def classification_loss(count):
#         if count >= 3000:
#             cls_id = int(count / 600)
#             cls_id = np.random.randint(0, cls_id, 5)
#             fake_sample = np.random.randint(0, 5)
#             query_sample = np.random.randint(0, 600)
#             for count_train in range(0, 600, 10):
#                 fake_feature = []
#                 query_feature = []
#                 for i in range(5):
#                     encoder.train()
#                     decoder.train()
#                     train_fake = feature_train[cls_id[i] * 600:cls_id[i] * 600 + 600]
#                     train_fake_pro = train_fake.mean(dim=0).repeat(5, 1)
#                     noise_fake = encoder(train_fake[count_train:count_train + 5], train_fake_pro)
#                     fake_feature.append(decoder(train_fake_pro, noise_fake)[fake_sample])
#                     query_feature.append(train_fake[query_sample])
#                 fake_feature = torch.stack(fake_feature)
#                 query_feature = torch.stack(query_feature)
#                 train_cls_labels = [i + 1.0 for i in range(5)]
#                 loss_classification = 0.0
#                 for i in range(query_feature.shape[0]):
#                     loss_classification += -1 * 1 * torch.log(
#                         torch.exp(cos(query_feature[i:i + 1], fake_feature[i:i + 1])) / torch.sum(torch.stack(
#                             [torch.exp(cos(query_feature[i:i + 1], fake_feature[j:j + 1])) for j in
#                              range(fake_feature.shape[0])])))
#                 loss_classification = (loss_classification / query_feature.shape[0])  # 上次是0.01、0.05（这个更好）,0.075
#                 return loss_classification.item()
#         else:
#             return 0


def adjust_learning_rate(optimizers, lr, dec_lr, iter):
    new_lr = lr * (0.5 ** (int(iter / dec_lr)))

    for optimizer in optimizers:
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr

if __name__ == '__main__':
# ---- data loading
    n_shot = 5
    n_ways = 80
    n_queries = 595
    n_runs = 1
    n_lsamples = n_ways * n_shot
    n_usamples = n_ways * n_queries
    n_samples = n_lsamples + n_usamples


    log = log_creater('./log/mini_1/trans/lr')

    import FSLTask
    cfg = {'shot':n_shot, 'ways':n_ways, 'queries':n_queries}
    base_datas = FSLTask.return_DataSet("miniimagenet_base")# 传入新类特征 data(20,600,640)
    val_datas = FSLTask.return_DataSet("miniimagenet_val")
    total_datas = torch.cat((base_datas, val_datas), dim=0).cuda()#（80,600,640）

    FSLTask.chang_train_data(total_datas)
    # total_datas = total_datas.reshape(-1, 640)#(48000,640)
    total_datas = FSLTask.GenerateRunSet(cfg=cfg)#(1,80,600,640)
    total_datas = total_datas.permute(0, 2, 1, 3).reshape(n_runs, n_samples, -1)
    train_labels = torch.arange(n_ways).view(1, 1, n_ways).expand(n_runs, n_shot+n_queries, 80).clone().view(n_runs, n_samples)
    # labels = torch.arange(5).view(1, 1, 5).expand(3, 20, 5).clone().view(3, 100)#(3,100)
    # labels = labels.permute(1, 0)

    # ndata = torch.squeeze(ndata)
    #
    # ndata = ndata.chunk(16, dim=0)
    # features = torch.arange(0, 64000).reshape(1, 100, 640).type(torch.FloatTensor)
    # for i in range(16):
    #     feature = ndata[i].chunk(30, dim=1)#30*(5,20)
    #     for j in range(30):
    #         feat = feature[j].permute(1, 0, 2).reshape(1, 100, -1)
    #         features = torch.cat((features, feat), dim=0)
    # total_datas = features[1:]#(480,100,640)

# Power transform

    sns.distplot(total_datas[0][0], hist=False, color='r', label='class 1')
    sns.distplot(total_datas[0][1], hist=False, color='g', label='class 2')
    sns.distplot(total_datas[0][2], hist=False, color='y', label='class 3')
    sns.distplot(total_datas[0][3], hist=False, color='b', label='class 4')
    sns.distplot(total_datas[0][4], hist=False, color='#FFB6C1', label='class 5')
    plt.title("miniImageNet")
    plt.legend()
    plt.show()
    beta = 0.5
    total_datas[:, ] = torch.pow(total_datas[:, ]+1e-6, beta)


    n_nfeat = total_datas.size(2)
    total_datas = scaleEachUnitaryDatas(total_datas)#除以2范式

    # # trans-mean-sub均值减法
    #
    total_datas = centerDatas(total_datas)

    sns.distplot(total_datas[0][0], hist=False, color='r', label='class 1')
    sns.distplot(total_datas[0][1], hist=False, color='g', label='class 2')
    sns.distplot(total_datas[0][2], hist=False, color='y', label='class 3')
    sns.distplot(total_datas[0][3], hist=False, color='b', label='class 4')
    sns.distplot(total_datas[0][4], hist=False, color='#FFB6C1', label='class 5')
    plt.title("miniImageNet")
    plt.legend()
    plt.show()

    total_datas = torch.squeeze(total_datas)  # (48000,640)
    total_datas = total_datas.chunk(600, dim=0)
    #
    # # total_datas = total_datas[:600].cpu()
    # total_datas = preprocessing.PowerTransformer(total_datas[:, ]).method.reshape(1, 48000, 640)
    # # plt.figure()
    # # sns.set()
    # # sns.distplot(total_datas)
    # # plt.show()
    # total_datas = QRreduction(total_datas)
    # total_datas = centerDatas(total_datas)
    # total_datas = torch.squeeze(total_datas)  # (48000,640)
    # total_datas = total_datas.type(torch.FloatTensor).cuda().chunk(600, dim=0)



    feature_train = torch.arange(640.0).reshape(1, 640).type(torch.FloatTensor).cuda()
    feature_train_3 = torch.arange(384000.0).reshape(1, 600, 640).type(torch.FloatTensor).cuda()
    train_pro = torch.arange(640.0).reshape(1, 640).type(torch.FloatTensor).cuda()

    for i in range(80):
        feature_train_2 = torch.arange(640.0).reshape(1, 640).type(torch.FloatTensor).cuda()
        for j in range(600):
            feature_train = torch.cat((feature_train, total_datas[j][i].reshape(1, -1).cuda()), dim=0)
            feature_train_2 = torch.cat((feature_train_2, total_datas[j][i].reshape(1, -1).cuda()), dim=0)
        query = torch.unsqueeze(feature_train_2[1:], dim=0)
        query = torch.unsqueeze(query, dim=2)
        query = query.transpose(1, 2)
        multihead_attn = ScaledDotProductAttention(temperature=query.size(2))
        attn_output = multihead_attn(query, query, query).squeeze(dim=0)
        query = query.squeeze()
        # train_pro = torch.matmul(attn_output, query)
        feature_train_3 = torch.cat((feature_train_3, feature_train_2[1:].unsqueeze(dim=0)), dim=0)
        train_pro = torch.cat((train_pro, torch.matmul(attn_output, query)), dim=0)
    feature_train = feature_train[1:]
    train_pro = train_pro[1:]
    feature_train_3 = feature_train_3[1:]
    torch.cuda.empty_cache()



    FSLTask.chang_train_data(feature_train_3)
    cfg = {'shot':24, 'ways':5, 'queries':1}
    train_feature_runs, class_runs = FSLTask.GenerateRunSet_and_pro(end=20000, cfg=cfg)
    train_feature_runs = train_feature_runs.cuda()
    torch.cuda.empty_cache()
    encoder_size = [5120]
    decoder_size = [5210]
    noise_size = 10
    drop_out_rate = 0.5
    drop_out_rate_input = 0.0
    learning_rate = 1e-3
    batch_size = 120
    decay_factor = 0.9
    num_epoch = 100
    encoder = Encoder(640, encoder_size, noise_size, drop_out_rate, drop_out_rate_input).cuda()
    decoder = Decoder(640, decoder_size, noise_size, drop_out_rate).cuda()
    optimizer = optimizer(encoder, decoder, lr=learning_rate, decay=1e-6)

    n_shot = 1
    n_ways = 5
    n_queries = 15
    n_runs = 10000
    n_lsamples = n_ways * n_shot
    n_usamples = n_ways * n_queries
    n_samples = n_lsamples + n_usamples

    cfg = {'shot': n_shot, 'ways': n_ways, 'queries': n_queries}
    FSLTask.loadDataSet("miniimagenet")  # 传入新类特征 data(20,600,640)
    FSLTask.setRandomStates(cfg, n_runs)
    ndatas = FSLTask.GenerateRunSet(end=n_runs, cfg=cfg)  # (10000,5,20,640)
    ndatas = ndatas.permute(0, 2, 1, 3).reshape(n_runs, n_samples, -1)  # (10000,100,640)
    labels = torch.arange(n_ways).view(1, 1, n_ways).expand(n_runs, n_shot + n_queries, 5).clone().view(n_runs,
                                                                                                        n_samples)  # (10000,100)
    # Power transform
    ndatas[:, ] = torch.pow(ndatas[:, ] + 1e-6, beta)
    # ndatas = QRreduction(ndatas)  # 特征维度转换成100维
    n_nfeat = ndatas.size(2)
    ndatas = scaleEachUnitaryDatas(ndatas)  # 除以2范式
    #
    # # trans-mean-sub均值减法
    #
    ndatas = centerDatas(ndatas)

    # ndatas = ndatas.reshape(-1, n_nfeat)
    # ndatas = preprocessing.PowerTransformer(ndatas[:, ]).method.reshape(n_runs, n_samples, -1).type(torch.FloatTensor)
    # ndatas = QRreduction(ndatas)
    # # print("size of the datas...", ndatas.size())  # （10000,100，100）
    # ndatas = centerDatas(ndatas)
    # switch to cuda
    ndatas = ndatas.cuda()
    labels = labels.cuda()
    feat_dim = feature_train.size(1)

    best_acc = 0.0
    last_file_name = ''
    last_loss_epoch = None
    lam = 10
    model = GaussianModel(n_ways, lam)
    model.initFromLabelledDatas()
    torch.cuda.empty_cache()
    alpha = 0.2
    optim = MAP(alpha)

    optim.verbose = True
    optim.progressBar = True
    torch.cuda.empty_cache()
    pro = optim.loop(model, n_epochs=20)




for run in range(train_feature_runs.size(0)):
        torch.cuda.empty_cache()
        encoder.train()
        decoder.train()
        train_feature_run = train_feature_runs[run][:, :24, :].reshape(-1, train_feature_runs.size(3))
        class_pro = class_runs[run]
        pro_2 = torch.zeros(5, 24, 640)
        for i in range(5):
            pro_2[i] = train_pro[class_pro[i]].repeat(24, 1)
        pro_2 = pro_2.reshape(-1, 640).cuda()
        loss_e, fake_features = loss(train_feature_run, pro_2)
        query_features = train_feature_runs[run][:, 24:, :].squeeze()
        fake_features = fake_features.reshape(5, -1, 640)
        loss_cls_total = 0.0
        torch.cuda.empty_cache()
        for i in range(fake_features.size(1)):
            fake_feature = fake_features[:, i, :]
            loss_classification = 0.0
            for i in range(query_features.shape[0]):
                torch.cuda.empty_cache()
                loss_classification += -1 * 1 * torch.log(
                    torch.exp(cos(query_features[i:i + 1], fake_feature[i:i + 1])) / torch.sum(torch.stack(
                        [torch.exp(cos(query_features[i:i + 1], fake_feature[j:j + 1])) for j in
                         range(fake_feature.shape[0])])))
            loss_classification = (loss_classification / query_features.shape[0])
            loss_cls_total += loss_classification
        loss_cls_total = loss_cls_total/fake_features.size(1)
        loss_total = loss_e+loss_cls_total
        torch.cuda.empty_cache()

        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()
        if run % 100 == 0:
            log.info(
                'The run-----: {} :------Total loss--------------:{}'.format(
                    run, loss_total.item()))

        adjust_learning_rate(optimizers=[optimizer],
                                  lr=learning_rate,
                                  iter=run,
                             dec_lr=2000)

        if run % 1000 == 0 and run > 0:

            Accuracy = []
            for runs in range(20):
                logits = euclidean_metric(pro[runs], train_pro)
                new_feature_train = torch.arange(384000.0).reshape(600, 640).cuda()
                for j in range(5):
                    torch.cuda.empty_cache()
                    class_id = logits[j].argmax().item()
                    # print(ndatas.shape)#(100,3000,640),5*600=3000,100runs
                    # print(feature_train.shape)#(48000,640)
                    shot_features = ndatas[runs][:n_shot * n_ways]
                    class_train = feature_train[class_id * 600:class_id * 600 + 600]
                    # for count in range(0, class_train.shape[0], batch_size):
                    #     encoder.train()
                    #     decoder.train()
                    #     data_batch = next_batch(feature_train[class_id * 600:class_id * 600 + 600], labels_train[class_id * 600:class_id * 600 + 600], count, count + batch_size).cuda()
                    #     proto_data = class_train.mean(dim=0).repeat(data_batch.shape[0], 1)
                    #
                    #     loss_e = loss(data_batch, proto_data)
                    #     optimizer.zero_grad()
                    #     loss_e.backward()
                    #     optimizer.step()
                    proty = train_pro[class_id].repeat(600, 1)
                    encoder.eval()
                    decoder.eval()
                    noise = encoder(class_train, proty)
                    new_feature = decoder(pro[runs][j].repeat(600, 1), noise)
                    new_feature_train = torch.cat((new_feature_train, new_feature), dim=0)
                new_feature_train = torch.cat((new_feature_train[600:], shot_features), dim=0)
                label = [0 for i in range(3005)]
                j = -1.0
                for i in range(3000):
                    if i % 600 == 0:
                        j = j + 1.0
                    label[i] = j
                for i in range(3000, 3005):
                    label[i] = i % 5
                classifier = LogisticRegression(max_iter=1000).fit(X=new_feature_train.cpu().detach().numpy(), y=label)

                predicts = classifier.predict(ndatas[runs][5:].cpu().detach().numpy())
                accQ = np.mean(predicts == labels[runs][5:].cpu().detach().numpy())
                # clf = SVC(kernel='linear')
                # clf.fit(X=new_feature_train.cpu().detach().numpy(), y=label)
                #
                # predicts = clf.predict(ndatas[runs][5:].cpu().detach().numpy())
                #
                # accQ = np.mean(predicts == labels[runs][5:].cpu().detach().numpy())
                Accuracy.append(accQ * 100)
                log.info('--- Accuracy on Query set : {}'.format(accQ*100))
                log.info('-----------------The run:{}-----the avg Accracy:{}-----'.format(runs, sum(Accuracy)/len(Accuracy)))
            Accuracy = torch.tensor(Accuracy)
            acc = Accuracy.mean()
            acc_los = Accuracy.std().item() * 1.96 / math.sqrt(runs)
            log.info(
                '---------------------------Total Accuracy-------------------- accuracy:{:0.2f} +- {:0.2f}'.format(
                    acc, acc_los))
            if acc > best_acc:
                if best_acc != 0.0:
                    os.remove(last_file_name)
                best_acc = acc
                last_file_name = os.getcwd() + "/model_weights/" + 'mini_trans_lr' + '_' \
                                 + str(n_shot) + '_shot_' \
                                 + str(run) + '_run_' \
                                 + str(np.around(best_acc.item(), decimals=4)) +'+_'+str(np.around(acc_los, decimals=4))+ '_acc.ckpt'
                save_model(encoder, decoder, last_file_name)


torch.cuda.empty_cache()
checkpoints = torch.load(last_file_name)
encoder.load_state_dict(checkpoints['encoder_state_dict'])
decoder.load_state_dict(checkpoints['decoder_state_dict'])
encoder.eval()
decoder.eval()
print('-----model loading success----------------')
Accuracy = []
for runs in range(10000):
    torch.cuda.empty_cache()
    logits = euclidean_metric(pro[runs], train_pro)
    new_feature_train = torch.arange(384000.0).reshape(600, 640).cuda()
    for j in range(5):
        torch.cuda.empty_cache()
        class_id = logits[j].argmax().item()
        shot_features = ndatas[runs][:n_shot * n_ways]
        class_train = feature_train[class_id * 600:class_id * 600 + 600]
        proty = train_pro[class_id].repeat(600, 1)
        noise = encoder(class_train, proty)
        new_feature = decoder(pro[runs][j].repeat(600, 1), noise)
        new_feature_train = torch.cat((new_feature_train, new_feature), dim=0)
    new_feature_train = torch.cat((new_feature_train[600:], shot_features), dim=0)
    label = [0 for i in range(3005)]
    j = -1.0
    for i in range(3000):
        if i % 600 == 0:
            j = j + 1.0
        label[i] = j
    for i in range(3000, 3005):
        label[i] = i % 5
    classifier = LogisticRegression(max_iter=1000).fit(X=new_feature_train.cpu().detach().numpy(), y=label)

    predicts = classifier.predict(ndatas[runs][5:].cpu().detach().numpy())
    accQ = np.mean(predicts == labels[runs][5:].cpu().detach().numpy())
    # clf = SVC(kernel='linear')
    # clf.fit(X=new_feature_train.cpu().detach().numpy(), y=label)
    #
    # predicts = clf.predict(ndatas[runs][5:].cpu().detach().numpy())

    # accQ = np.mean(predicts == labels[runs][5:].cpu().detach().numpy())
    Accuracy.append(accQ*100)
    log.info('--- Accuracy on Query set : {}'.format(accQ*100))
    log.info('-----------------The run:{}-----the avg Accracy:{}-----'.format(runs, sum(Accuracy)/len(Accuracy)))
Accuracy = torch.tensor(Accuracy)
acc = Accuracy.mean()
acc_los = Accuracy.std().item() * 1.96 / math.sqrt(runs)
log.info(
    '---------------------------Total Accuracy-------------------- accuracy:{:0.2f} +- {:0.2f}'.format(
        acc, acc_los))
