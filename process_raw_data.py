#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import pickle
import pandas as pd

base_dir = os.getcwd()

# 原始数据
atis_raw_train = os.path.join(base_dir, 'atis_new', 'atis.train.pkl')
atis_raw_test = os.path.join(base_dir, 'atis_new', 'atis.test.pkl')

# 处理后保存为csv文件
atis_train_csv = os.path.join(base_dir, 'atis_new', 'atis.train.csv')
atis_test_csv = os.path.join(base_dir, 'atis_new', 'atis.test.csv')


def load_atis(file_path):
    with open(file_path, 'rb') as f_read:
        ds,dicts = pickle.load(f_read)
    print('done loading:', file_path)
    print('samples:{}'.format(len(ds['query'])))
    print('vocab_size:{}'.format(len(dicts['token_ids'])))
    print('slot count:{}'.format(len(dicts['slot_ids'])))
    print('intent count:{}'.format(len(dicts['intent_ids'])))
    return ds, dicts


'''
    处理训练数据，保存为csv结构
'''
train_ds, train_dicts = load_atis(atis_raw_train)

t2i, s2i, in2i = map(train_dicts.get, ['token_ids', 'slot_ids', 'intent_ids'])
i2t, i2s, i2in = map(lambda d:{d[k]:k for k in d.keys()}, [t2i, s2i, in2i])

query, slots, intent = map(train_ds.get, ['query', 'slot_labels', 'intent_labels'])

train_source_target = []
for i in range(len(train_ds['query'])):
    intent_source_target_lst = []

    # 1.存储intent
    intent_source_target_lst.append(i2in[intent[i][0]])

    # 2.存储source
    source_data = list(' '.join(map(i2t.get, query[i])).split())
    # 删除BOS
    del(source_data[0])
    # 删除EOS
    del(source_data[-1])
    intent_source_target_lst.append(' '.join(source_data))

    # 3.存储target
    target_data = [i2s[slots[i][j]] for j in range(len(query[i]))]
    # 删除BOS
    del(target_data[0])
    # 删除EOS
    del(target_data[-1])
    intent_source_target_lst.append(' '.join(target_data))

    train_source_target.append(intent_source_target_lst)

name = ['intent', 'source', 'target']

train_csv = pd.DataFrame(columns=name, data=train_source_target)
train_csv.to_csv(atis_train_csv)

print('train data process done!')


'''
    处理测试数据，保存为csv结构
'''
test_ds, test_dicts = load_atis(atis_raw_test)

t2i, s2i, in2i = map(test_dicts.get, ['token_ids', 'slot_ids', 'intent_ids'])
i2t, i2s, i2in = map(lambda d:{d[k]:k for k in d.keys()}, [t2i, s2i, in2i])

query, slots, intent = map(test_ds.get, ['query', 'slot_labels', 'intent_labels'])

test_source_target = []
for i in range(len(test_ds['query'])):
    intent_source_target_lst = []

    # 1.存储intent
    intent_source_target_lst.append(i2in[intent[i][0]])

    # 2.存储source
    source_data = list(' '.join(map(i2t.get, query[i])).split())
    # 删除BOS
    del(source_data[0])
    # 删除EOS
    del(source_data[-1])
    intent_source_target_lst.append(' '.join(source_data))

    # 3.存储target
    target_data = [i2s[slots[i][j]] for j in range(len(query[i]))]
    # 删除BOS
    del(target_data[0])
    # 删除EOS
    del(target_data[-1])
    intent_source_target_lst.append(' '.join(target_data))

    test_source_target.append(intent_source_target_lst)

name = ['intent', 'source', 'target']

test_csv = pd.DataFrame(columns=name, data=test_source_target)
test_csv.to_csv(atis_test_csv)

print('test data process done!')


pd_train = pd.read_csv(atis_train_csv, index_col=0)
pd_train.head()
