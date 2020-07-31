# -*- coding: utf-8 -*-#
# Name:         make_tf_records_zh
# Author:       ARCHI
# Date:         2020/7/31
# Description:  中文预料的预处理脚本
# 将TXT语料转换为tf-records格式的文件
# -------------------------------------------------------------------------------
import numpy as np
import os
import tensorflow as tf
import tqdm
import pdb
import glob
import time
import sys
import re
import argparse
import fastBPE
import platform

use_py3 = platform.python_version()[0] == '3'

parser = argparse.ArgumentParser(description='TensorFlow code for creating TFRecords data')
parser.add_argument('--text_file',
                    type=str, required=True,
                    help='location of text file to convert to TFRecords')
parser.add_argument('--control_code',
                    type=str, required=True,
                    help='control code to use for this file. must be in the vocabulary, else it will error out.')
parser.add_argument('--sequence_len',
                    type=int, required=True,
                    help='sequence length of model being fine-tuned (256 or 512)')

args = parser.parse_args()

path_to_train_file = fname = args.text_file
domain = [args.control_code]

train_text = open(path_to_train_file, 'rb').read().decode(encoding='utf-8')
bpe = fastBPE.fastBPE('../codes', '../vocab')
# will NOT work for non-English texts
tokenized_train_text = bpe.apply([train_text.encode('ascii', errors='ignore') if not use_py3 else train_text])[0]
# if you want to run non-english text, please tokenize separately using ./fast applybpe and then run this script on the .bpe file with utf8 encoding

tokenized_train_text = re.findall(r'\S+|\n', tokenized_train_text)
tokenized_train_text = list(filter(lambda x: x != u'@@', tokenized_train_text))

# load the vocabulary from file
if not use_py3:
    vocab = open('../vocab').read().decode(encoding='utf-8').split('\n')
else:
    vocab = open('../vocab', encoding='utf-8').read().split('\n')

vocab = list(map(lambda x: x.split(' ')[0], vocab)) + ['<unk>'] + ['\n']
print('{} unique words'.format(len(vocab)))

if args.control_code not in vocab:
    print('Provided control code is not in the vocabulary')
    print('Please provide a different one; refer to the vocab file for allowable tokens')
    sys.exit(1)

# Creating a mapping from unique characters to indices
word2idx = {u: i for i, u in enumerate(vocab)}
idx2word = np.array(vocab)

seq_length = args.sequence_len - 1


# 对字符集中的字符进行寻址（vocab中的index），对于OOV字符统一按<unk>处理
def numericalize(x):
    count = 0
    for i in x:
        if i not in word2idx:
            print(i)
            count += 1
    return count > 1, [word2idx.get(i, word2idx['<unk>']) for i in x]


"""
滑窗形式的构建输入和输出数据
原始文本：[word1, word2, ... word_n]
构造步骤：
    1. 循环截取长度为k(sequence_length - 1)的词序列 [word_i, word_i+1, word_i+2, ... word_i+k-1]
    2. 添加控制标识[control_word, word_i, word_i+1, word_i+2, ... word_i+k-1],并映射为idx序列（作为输入）
    3. 当前窗口右滑一个单位，取 [word_i+1, word_i+2, ... word_i+k]，并映射为idx序列（作为模型输出）
    4. 循环执行步骤 1,2,3
"""
total = 0
skipped = 0
with tf.io.TFRecordWriter(fname.lower() + '.tfrecords') as writer:
    for i in tqdm.tqdm(range(0, len(tokenized_train_text), seq_length)):
        flag_input, inputs = numericalize(domain + tokenized_train_text[i:i + seq_length])
        flag_output, outputs = numericalize(tokenized_train_text[i:i + seq_length + 1])
        total += 1
        if flag_input or flag_output:
            skipped += 1
            continue

        if len(inputs) != seq_length + 1 or len(outputs) != seq_length + 1:
            break
        example_proto = tf.train.Example(
            features=tf.train.Features(
                feature={'input': tf.train.Feature(int64_list=tf.train.Int64List(value=inputs)),
                         'output': tf.train.Feature(int64_list=tf.train.Int64List(value=outputs))}))
        writer.write(example_proto.SerializeToString())
print('Done')
print('Skipped {} of {}'.format(skipped, total))
