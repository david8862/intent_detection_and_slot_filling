#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import numpy as np
import os
import json
import random

# load source words
source_words_path = os.path.join(os.getcwd(), 'source_words.pkl')
with open(source_words_path, 'rb') as f_source_words:
    source_words = pickle.load(f_source_words)

# load target words
target_words_path = os.path.join(os.getcwd(), 'target_words.pkl')
with open(target_words_path, 'rb') as f_target_words:
    target_words = pickle.load(f_target_words)

# load label words
label_words_path = os.path.join(os.getcwd(), 'label_words.pkl')
with open(label_words_path, 'rb') as f_label_words:
    label_words = pickle.load(f_label_words)

print(source_words.__dict__.keys())

print(len(source_words))
print(len(target_words))
print(len(label_words))
print(source_words.stoi['<pad>'])
print(source_words.stoi['<eos>'])
print(source_words.stoi['<sos>'])
print(source_words.stoi['<unk>'])
print(target_words.stoi['<pad>'])


'''
编码器Encoder的实现
'''
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, intent_dim, hid_dim, n_layers, kernel_size, dropout, max_length=50):
        super(Encoder, self).__init__()

        assert kernel_size % 2 == 1,'kernel size must be odd!' # 卷积核size为奇数，方便序列两边pad处理

        self.scale = torch.sqrt(torch.FloatTensor([0.5])).to(device) # 确保整个网络的方差不会发生显著变化

        self.tok_embedding = nn.Embedding(input_dim, emb_dim) # token编码
        self.pos_embedding = nn.Embedding(max_length, emb_dim) # token的位置编码

        self.emb2hid = nn.Linear(emb_dim, hid_dim) # 线性层，从emb_dim转为hid_dim
        self.hid2emb = nn.Linear(hid_dim, emb_dim) # 线性层，从hid_dim转为emb_dim

        # 卷积块
        self.convs = nn.ModuleList([nn.Conv1d(in_channels=hid_dim,
                                              out_channels=2*hid_dim, # 卷积后输出的维度，这里2*hid_dim是为了后面的glu激活函数
                                              kernel_size=kernel_size,
                                              padding=(kernel_size - 1)//2) # 序列两边补0个数，保持维度不变
                                              for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)

        # 利用encoder的输出进行意图识别
        self.intent_output = nn.Linear(emb_dim, intent_dim)

    def forward(self, src):
        # src: [batch_size, src_len]
        batch_size = src.shape[0]
        src_len = src.shape[1]

        # 创建token位置信息
        pos = torch.arange(src_len).unsqueeze(0).repeat(batch_size, 1).to(device) # [batch_size, src_len]

        # 对token与其位置进行编码
        tok_embedded = self.tok_embedding(src) # [batch_size, src_len, emb_dim]
        pos_embedded = self.pos_embedding(pos.long()) # [batch_size, src_len, emb_dim]

        # 对token embedded和pos_embedded逐元素加和
        embedded = self.dropout(tok_embedded + pos_embedded) # [batch_size, src_len, emb_dim]

        # embedded经过一线性层，将emb_dim转为hid_dim，作为卷积块的输入
        conv_input = self.emb2hid(embedded) # [batch_size, src_len, hid_dim]

        # 转变维度，卷积在输入数据的最后一维进行
        conv_input = conv_input.permute(0, 2, 1) # [batch_size, hid_dim, src_len]

        # 以下进行卷积块
        for i, conv in enumerate(self.convs):
            # 进行卷积
            conved = conv(self.dropout(conv_input)) # [batch_size, 2*hid_dim, src_len]

            # 进行激活glu
            conved = F.glu(conved, dim=1) # [batch_size, hid_dim, src_len]

            # 进行残差连接
            conved = (conved + conv_input) * self.scale # [batch_size, hid_dim, src_len]

            # 作为下一个卷积块的输入
            conv_input = conved

        # 经过一线性层，将hid_dim转为emb_dim，作为enocder的卷积输出的特征
        conved = self.hid2emb(conved.permute(0, 2, 1)) # [batch_size, src_len, emb_dim]

        # 又是一个残差连接，逐元素加和输出，作为encoder的联合输出特征
        combined = (conved + embedded) * self.scale # [batch_size, src_len, emb_dim]

        # 意图识别,加一个平均池化,池化后的维度是：[batch_size, emb_dim]
        intent_output = self.intent_output(F.avg_pool1d(combined.permute(0, 2, 1), combined.shape[1]).squeeze()) # [batch_size, intent_dim]

        return conved, combined, intent_output

'''
解码器Decoder实现
'''
class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers,kernel_size, dropout, trg_pad_idx, max_length=50):
        super(Decoder, self).__init__()
        self.kernel_size = kernel_size
        self.trg_pad_idx = trg_pad_idx

        self.scale = torch.sqrt(torch.FloatTensor([0.5])).to(device)

        self.tok_embedding = nn.Embedding(output_dim, emb_dim)
        self.pos_embedding = nn.Embedding(max_length, emb_dim)

        self.emb2hid = nn.Linear(emb_dim, hid_dim)
        self.hid2emb = nn.Linear(hid_dim, emb_dim)

        self.attn_hid2emb = nn.Linear(hid_dim, emb_dim)
        self.attn_emb2hid = nn.Linear(emb_dim, hid_dim)

        self.fc_out = nn.Linear(emb_dim, output_dim)

        self.convs = nn.ModuleList([nn.Conv1d(in_channels=hid_dim,
                                              out_channels=2*hid_dim,
                                              kernel_size=kernel_size)
                                              for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)

    def calculate_attention(self, embedded, conved, encoder_conved, encoder_combined):
        '''
        embedded:[batch_size, trg_Len, emb_dim]
        conved:[batch_size, hid_dim, trg_len]
        encoder_conved:[batch_size, src_len, emb_dim]
        encoder_combined:[batch_size, src_len, emb_dim]
        '''
        # 经过一线性层，将hid_dim转为emb_dim，作为deocder的卷积输出的特征
        conved_emb = self.attn_hid2emb(conved.permute(0, 2, 1)) # [batch_size, trg_len, emb_dim]

        # 一个残差连接，逐元素加和输出，作为decoder的联合输出特征
        combined = (conved_emb + embedded) * self.scale # [batch_size, trg_len, emb_dim]

        # decoder的联合特征combined与encoder的卷积输出进行矩阵相乘
        energy = torch.matmul(combined, encoder_conved.permute(0, 2, 1)) # [batch_size, trg_len, src_len]

        attention = F.softmax(energy, dim=2) # [batch_size, trg_len, src_len]

        attention_encoding = torch.matmul(attention, encoder_combined) # [batch_size, trg_len, emb_dim]

        # 经过一线性层，将emb_dim转为hid_dim
        attended_encoding = self.attn_emb2hid(attention_encoding) # [batch_size, trg_len, hid_dim]

        # 一个残差连接，逐元素加和输出
        attended_combined = (conved + attended_encoding.permute(0, 2, 1)) * self.scale # [batch_size, hid_dim, trg_len]

        return attention, attended_combined

    def forward(self, trg, encoder_conved, encoder_combined):
        '''
        trg:[batch_size, trg_len]
        encoder_conved:[batch_size, src_len, emb_dim]
        encoder_combined:[batch_size, src_len, emb_dim]
        '''
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]

        # 位置编码
        pos = torch.arange(trg_len).unsqueeze(0).repeat(batch_size, 1).to(device) # [batch_size, trg_len]

        # 对token和pos进行embedding
        tok_embedded = self.tok_embedding(trg) # [batch_size, trg_len, emb_dim]
        pos_embedded = self.pos_embedding(pos.long()) # [batch_size, trg_len, emb_dim]

        # 对token embedded和pos_embedded逐元素加和
        embedded = self.dropout(tok_embedded + pos_embedded) # [batch_size, trg_len, emb_dim]

        # 经过一线性层，将emb_dim转为hid_dim，作为卷积的输入
        conv_input = self.emb2hid(embedded) # [batch_size, trg_len, hid_dim]

        # 转变维度，卷积在输入数据的最后一维进行
        conv_input = conv_input.permute(0, 2, 1) # [batch_size, hid_dim, trg_len]

        batch_size = conv_input.shape[0]
        hid_dim = conv_input.shape[1]

        # 卷积块
        for i, conv in enumerate(self.convs):
            conv_input = self.dropout(conv_input)

            # 在序列的一端进行pad
            padding = torch.zeros(batch_size, hid_dim, self.kernel_size - 1).fill_(self.trg_pad_idx).to(device)

            padded_conv_input = torch.cat((padding, conv_input), dim=2) # [batch_size, hid_dim, trg_len + kernel_size - 1]

            # 进行卷积
            conved = conv(padded_conv_input) # [batch_size, 2 * hid_dim, trg_len]

            # 经过glu激活
            conved = F.glu(conved, dim=1) # [batch_size, hid_dim, trg_len]

            # 计算attention
            attention, conved = self.calculate_attention(embedded, conved, encoder_conved, encoder_combined) # [batch_size, trg_len, src_len], [batch_size, hid_dim, trg_len]

            # 残差连接
            conved = (conved + conv_input) * self.scale # [batch_size, hid_dim, trg_len]

            # 作为下一层卷积的输入
            conv_input = conved

        conved = self.hid2emb(conved.permute(0, 2, 1)) # [batch_size, trg_len, emb_dim]

        # 预测输出
        output = self.fc_out(self.dropout(conved)) # [batch_size, trg_len, output_dim]

        return output, attention

# 包装Encoder与Decoer
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()

        # 编码器
        self.encoder = encoder

        # 解码器用于slot槽识别
        self.decoder = decoder

    def forward(self, src, trg):
        '''
        src:[batch_size, src_len]
        trg:[batch_size, trg_Len-1] # decoder的输入去除了<eos>

        encoder_conved是encoder中最后一个卷积层的输出
        encoder_combined是encoder_conved + (src_embedding + postional_embedding)
        '''
        encoder_conved, encoder_combined, intent_output = self.encoder(src) # [batch_size, src_len, emb_dim]; [batch_size, src_len, emb_dim]

        # decoder是对一批数据进行预测输出
        slot_output, attention = self.decoder(trg, encoder_conved, encoder_combined) # [batch_size, trg_len-1, output_dim]; [batch_size, trg_len-1, src_len]

        return intent_output, slot_output, attention


emb_dim = 64
hid_dim = 32
enc_layers = 5
dec_layers = 5
enc_kernel_size = 3
dec_kernel_size = 3
enc_dropout = 0.25
dec_dropout = 0.25

model_path = os.path.join(os.getcwd(), "model.h5")

input_dim = len(source_words) # source 词典大小（即词数量）
output_dim = len(target_words) # target 词典大小（即实体类型数量）
intent_dim = len(label_words) # label 词典大小（即意图类别数量）
trg_pad_idx = target_words.stoi['<pad>']
print('trg_pad_idx:', trg_pad_idx)

enc = Encoder(input_dim, emb_dim, intent_dim, hid_dim, enc_layers, enc_kernel_size, enc_dropout)
dec = Decoder(output_dim, emb_dim, hid_dim, dec_layers, dec_kernel_size, dec_dropout, trg_pad_idx)

model = Seq2Seq(enc, dec).to(device)

model.load_state_dict(torch.load(model_path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")))


def predict_slot_intent(sentence, model):

    model.eval()

    tokenized = sentence.split()  # tokenize the sentence
    tokenized = ['<sos>'] + tokenized + ['<eos>']
    indexed = [source_words.stoi[t] for t in tokenized]  # convert to integer sequence

    print(tokenized)
    print(indexed)

    src_tensor = torch.LongTensor(indexed)  # convert to tensor
    src_tensor = src_tensor.unsqueeze(0).to(device)  # reshape in form of batch,no. of words

    with torch.no_grad():
        encoder_conved, encoder_combined, intent_output = model.encoder(src_tensor)
    intent_output = intent_output.squeeze()
    intent_output = intent_output.argmax()
    intent = intent_output.detach().item()

    intent_label = label_words.itos[intent]

    trg_indexes = [target_words.stoi['<sos>']]
    print('sequence length: {}'.format(src_tensor.shape[1]))
    for i in range(1, src_tensor.shape[1]):
        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)

        with torch.no_grad():
            slot_output, attention = model.decoder(trg_tensor, encoder_conved, encoder_combined)

        pred_token = slot_output.argmax(2)[:, -1].item()

        trg_indexes.append(pred_token)

    trg_tokens = [target_words.itos[i] for i in trg_indexes]

    print('slot prediction: {}'.format(trg_tokens[1:]))
    print('intent prediction: {}'.format(intent_label))


sentence = "i would like to find a flight from charlotte to las vegas that makes a stop in st. louis"
sentence2 = "which airlines have first class flights today"
sentence3 = "is there any taxi available in boston"

predict_slot_intent(sentence3, model)

