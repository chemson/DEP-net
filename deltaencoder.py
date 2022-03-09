import torch
import torch.nn as nn
import os



class Encoder(nn.Module):
    def __init__(self, feature_dim=256, encoder_size=[8192], z_dim=16, dropout=0.5, dropout_input=0.0, leak=0.2):
        super(Encoder, self).__init__()
        self.first_linear = nn.Linear(feature_dim*2, encoder_size[0])#全连接层（in_feature_size,out_feature_size)特征维度*2，输出四倍

        linear = []
        #print(encoder_size[0]) 8192
        #print(encoder_size) {list：1}存的值为8192
        for i in range(len(encoder_size) - 1):
            linear.append(nn.Linear(encoder_size[i], encoder_size[i+1]))
            linear.append(nn.LeakyReLU(leak))
            linear.append(nn.Dropout(dropout))

        self.linear = nn.Sequential(*linear)
        self.final_linear = nn.Linear(encoder_size[-1], z_dim)#全连接层，下标-1指的是最后一个数
        self.lrelu = nn.LeakyReLU(leak)#激活
        self.dropout_input = nn.Dropout(dropout_input)
        self.dropout = nn.Dropout(dropout)
        self.Tanh = nn.Tanh()
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim * 4),
            nn.Tanh(),
            nn.Linear(feature_dim * 4, z_dim),
            nn.Tanh(),
        )

    def forward(self, features, reference_features):#这里是张量，训练阶段：feature、reference_feature都是(128,2048)
        features = self.dropout_input(features)
        x = torch.cat([features, reference_features], 1)#在维度1对给定的张量进行拼接（128,4096）
        #x(1024,4096),feature/reference_features(1024,2048)
        #训练阶段：x(128,4096)
        # print("features shape is:", features.shape, reference_features.shape)
        # print(x.shape)

        x = self.first_linear(x)#(1024,8192)训练阶段：(128,8192)
        x = self.Tanh(x)#同上
        x = self.final_linear(x)#(1024,16)训练阶段：(128,16)
        # x = self.encoder(x)

        return x

class Decoder(nn.Module):
    def __init__(self, feature_dim=256, decoder_size=[8192], z_dim=16, dropout=0.5, leak=0.2):
        super(Decoder, self).__init__()
        self.first_linear = nn.Linear(z_dim+feature_dim, decoder_size[0])

        linear = []
        for i in range(len(decoder_size) - 1):
            linear.append(nn.Linear(decoder_size[i], decoder_size[i+1]))
            linear.append(nn.LeakyReLU(leak))
            linear.append(nn.Dropout(dropout))

        self.linear = nn.Sequential(*linear)

        self.final_linear = nn.Linear(decoder_size[-1], feature_dim)
        self.lrelu = nn.LeakyReLU(leak)
        self.dropout = nn.Dropout(dropout)
        self.Tanh = nn.Tanh()
        self.decoder = nn.Sequential(
            nn.Linear(z_dim+feature_dim, feature_dim*4),
            nn.Tanh(),
            nn.Linear(feature_dim*4, feature_dim),
            nn.Sigmoid()
        )

    def forward(self, reference_features, code):
        x = torch.cat([reference_features, code], 1)

        x = self.first_linear(x)
        x = self.Tanh(x)
        x = self.final_linear(x)

        # x = self.decoder(x)

        return x
# class Encoder(nn.Module):
#     def __int__(self, in_shape=640, out_shape=16, dropout=0.5):
#         super(Encoder, self).__int__()
#         self.in_shape = in_shape
#         self.out_shape = out_shape
#         self.Encoder = nn.Sequential(
#             nn.Linear(self.in_shape, 640),
#             nn.Tanh(),
#             nn.Linear(640, 320),
#             nn.Tanh(),
#             nn.Linear(320, 160),
#             nn.Tanh(),
#             nn.Linear(160, self.out_shape),
#             nn.Tanh()
#         )
#
#     def forward(self, reference_features, x):
#         x = torch.cat([reference_features, x], 1)
#         encoder = self.Encoder(x)
#         return encoder
#
#
# class Decoder(nn.Module):
#     def __int__(self, in_shape=16, out_shape=640, dropout=0.5):
#         super(Decoder, self).__int__()
#         self.in_shape = in_shape
#         self.out_shape = out_shape
#         self.Decoder = nn.Sequential(
#             nn.Linear(self.in_shape, 160),
#             nn.Tanh(),
#             nn.Linear(160, 320),
#             nn.Tanh(),
#             nn.Linear(320, self.out_shape),
#             nn.Sigmoid()
#         )
#
#     def forward(self, reference_features, x):
#         x = torch.cat([reference_features, x], 1)
#         decoder = self.Decoder(x)
#         return decoder