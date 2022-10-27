import os
import sys
import random
import argparse

import tensorflow.compat.v1 as tf

from src.configuration import ChatConfig

tf.disable_v2_behavior()
import numpy as np

from src.utils import matrix_initializer, truncated_normal_initializer_variable, zero_initializer_variable
from data_utils.prepare_dialogue_data import get_word_count, construct_word_dict, read_emotion_words, construct_vocab, \
    get_word_list, read_word_embeddings

__author__ = "Jocelyn"

# emotion_dict = {"anger": 0, "disgust": 1, "happiness": 2, "like": 3, "sadness": 4, "neutral": 5}
emotion_dict = {"anger": 0, "disgust": 1, "fear": 2, "joy": 3, "neutral": 4, "sadness": 5, "surprise": 6}
id2emotion = {idx: emotion for emotion, idx in emotion_dict.items()}
FLAGS = None


class LstmClassifier(object):
    def __init__(self, word_embeddings, words2idx, embedding_size, hidden_size, emotion_class, batch_size,
                 max_len, use_lstm, session, keep_prob=2.0, learning_rate=0.1, lr_decay=0.5, name="lstm_classifier"):
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.emotion_class = emotion_class
        self.batch_size = batch_size
        self.max_len = max_len
        self.use_lstm = use_lstm
        self.sess = session
        self.keep_prob = keep_prob
        self.learning_rate_decay_factor = lr_decay
        self.name = name

        # word embeddings
        self.words2idx = words2idx
        self.idx2words = {idx: word for word, idx in self.words2idx.items()}
        print("word embeddings size here")
        # print(np.shape(word_embeddings))
        self.embeddings = matrix_initializer(w=word_embeddings, name=self.name + "_word_embeddings")
        self.vocab_size = len(words2idx)

        # softmax
        self.sfx_w = truncated_normal_initializer_variable(width=hidden_size,
                                                           shape=[2 * self.hidden_size, self.emotion_class],
                                                           name=self.name+"_softmax_w")
        self.sfx_b = zero_initializer_variable(shape=[self.emotion_class], name=self.name+"_softmax_b")

        # placeholder
        self.input_x = tf.placeholder(shape=[self.batch_size, self.max_len], dtype=tf.int32, name=self.name+"_input_x")
        self.input_y = tf.placeholder(shape=[self.batch_size], dtype=tf.int32, name=self.name+"_input_y")
        self.input_len = tf.placeholder(shape=[self.batch_size], dtype=tf.int32, name=self.name+"_input_len")

        self.input_x_beam = tf.placeholder(shape=[self.batch_size,20, self.max_len], dtype=tf.int32,
                                      name=self.name + "_input_x_beam")
        self.input_len_beam = tf.placeholder(shape=[self.batch_size,20], dtype=tf.int32, name=self.name + "_input_len_beam")
        self.input_emo= tf.placeholder(shape=[self.batch_size,20],dtype=tf.int32, name=self.name+"_input_emo")


        self.forward_cell = self.rnn_cell()
        self.backward_cell = self.rnn_cell()

        # loss
        self.loss = self.compute_loss()
        self.pred_scores = self.predict_scores()
        self.pred_labels = tf.argmax(self.pred_scores, axis=1)
        self.beam_pred_labels= self.beam_fetch()

        tf.summary.scalar("loss", self.loss)

        self.global_step = tf.Variable(0, name=self.name + "_global_step", trainable=False)
        self.lr = tf.Variable(learning_rate, dtype=tf.float32,name=self.name + "_lr",trainable=False)
        self.train = self.optimize()

        self.sess.run(tf.global_variables_initializer())

        # self.variables_dict={'lstm_classifier_global_step':self.global_step,'lstm_classifier_softmax_b':self.sfx_b,'lstm_classifier_softmax_w':self.sfx_w,'lstm_classifier_word_embeddings':self.embeddings}
        self.saver = tf.train.Saver()

    def basic_rnn_cell(self):
        if self.use_lstm:
            return tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size, forget_bias=0.0, state_is_tuple=True,
                                                reuse=tf.get_variable_scope().reuse)
        else:
            return tf.nn.rnn_cell.GRUCell(self.hidden_size, reuse=tf.get_variable_scope().reuse)

    def rnn_cell(self):
        single_cell = self.basic_rnn_cell
        if self.keep_prob < 1.0:
            def single_cell():
                return tf.nn.rnn_cell.DropoutWrapper(single_cell, output_keep_prob=self.keep_prob)
        cell = single_cell()
        # cell = tf.nn.rnn_cell.MultiRNNCell([single_cell() for _ in range(self.num_layers)], state_is_tuple=True)
        return cell

    def lstm_process(self):
        input_embeddings = tf.nn.embedding_lookup(self.embeddings, self.input_x)  # [batch, max_time, embedding size]

        initiate_state_forward = self.forward_cell.zero_state(self.batch_size, dtype=tf.float32)
        initiate_state_backward = self.backward_cell.zero_state(self.batch_size, dtype=tf.float32)
        outputs, states = tf.nn.bidirectional_dynamic_rnn(self.forward_cell, self.backward_cell, input_embeddings,
                                                          sequence_length=self.input_len,
                                                          initial_state_fw=initiate_state_forward,
                                                          initial_state_bw=initiate_state_backward,
                                                          dtype=tf.float32)
        output_fw_state, output_bw_state = states
        final_states = tf.concat([output_fw_state, output_bw_state], axis=-1)  # [2, batch, 2 * hidden]
        split_states_outputs = tf.split(final_states, num_or_size_splits=2, axis=0)
        final_states = tf.reshape(split_states_outputs[1], [self.batch_size, 2 * self.hidden_size])
        """
                outputs, states = tf.nn.dynamic_rnn(self.forward_cell, input_embeddings, sequence_length=self.input_len,
                                                    initial_state=initiate_state_forward, dtype=tf.float32)
                split_states_outputs = tf.split(states, num_or_size_splits=2, axis=0)
                final_states = tf.reshape(split_states_outputs[1], [self.batch_size, self.hidden_size])
                """
        return final_states

    def lstm_process_beam(self):
        input_x_re=[]
        final_states_re=[]
        # 每个beam单独处理
        for i in range(20):
            input_x_inn = []
            input_len_inn=[]
            for j in range(self.batch_size):
                # input_x_inn形状是[batch_size,max_len]
                # input_len_beam形状就是[batch_size]
                input_x_inn.append(self.input_x_beam[j][i])
                input_len_inn.append(self.input_len_beam[j][i])
            # input_x_re形状就成了[beam,batch_size,max_len]
            print("input_len_beam here:")
            print(len(input_len_inn))
            input_x_re.append(input_x_inn)

            input_embeddings = tf.nn.embedding_lookup(self.embeddings,input_x_inn)  # [batch, max_time, embedding size]

            initiate_state_forward = self.forward_cell.zero_state(self.batch_size, dtype=tf.float32)
            initiate_state_backward = self.backward_cell.zero_state(self.batch_size, dtype=tf.float32)
            outputs, states = tf.nn.bidirectional_dynamic_rnn(self.forward_cell, self.backward_cell, input_embeddings,
                                                              sequence_length=input_len_inn,
                                                              initial_state_fw=initiate_state_forward,
                                                              initial_state_bw=initiate_state_backward,
                                                              dtype=tf.float32)
            output_fw_state, output_bw_state = states
            final_states = tf.concat([output_fw_state, output_bw_state], axis=-1)  # [2, batch, 2 * hidden]
            split_states_outputs = tf.split(final_states, num_or_size_splits=2, axis=0)
            final_states = tf.reshape(split_states_outputs[1], [self.batch_size, 2 * self.hidden_size])
            final_states_re.append(final_states)
        return final_states_re


    def compute_loss(self):
        final_states = self.lstm_process()
        logits = tf.matmul(final_states, self.sfx_w) + self.sfx_b
        entropy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_y, logits=logits)
        loss = tf.reduce_sum(entropy_loss) / self.batch_size
        return loss

    def predict_scores(self):
        final_states = self.lstm_process()
        logits = tf.matmul(final_states, self.sfx_w) + self.sfx_b
        scores = tf.nn.softmax(logits, dim=-1)
        return scores

    def beam_fetch(self):
        final_states=self.lstm_process_beam()
        scores_re=[]
        score_res_gro=[]
        for i in range(20):
            logits=tf.matmul(final_states[i],self.sfx_w)+self.sfx_b
            scores=tf.nn.softmax(logits,dim=-1)
            scores_re.append(scores)
        # score_re的形状：[beam,batch_size,total-7]
        for j in range(self.batch_size):
            score_res_col=[]
            for i in range(20):
                emo_index=self.input_emo[j][i]
                emo_tar=scores_re[i][j][emo_index]
                score_res_col.append(emo_tar)
                print('length of score_res_col')
                print(len(score_res_col))
            score_res_gro.append(score_res_col)


        return score_res_gro



    def optimize(self):
        optimizer = tf.train.GradientDescentOptimizer(self.lr)
        trainer = optimizer.minimize(self.loss)
        return trainer

    def train_step(self, this_input_x, this_input_y, this_input_len):
        output_feed = [self.train, self.loss]
        input_feed = {self.input_x: this_input_x,
                      self.input_y: this_input_y,
                      self.input_len: this_input_len}
        _, loss = self.sess.run(output_feed, input_feed)
        return loss

    def predict_step(self, this_input_x, this_input_len):
        output_feed = [self.pred_labels]
        input_feed = {self.input_x: this_input_x,
                      self.input_len: this_input_len}
        results = self.sess.run(output_feed, input_feed)
        return results[0]

    def beam_predict_step(self, this_input_x, this_input_len,this_input_emo):
        output_feed = [self.beam_pred_labels]
        input_feed = {self.input_x_beam: this_input_x,
                      self.input_len_beam: this_input_len,
                      self.input_emo:this_input_emo
                      }
        results = self.sess.run(output_feed, input_feed)
        return results[0]

    def get_train_batch(self, input_responses, input_labels, input_lengths, index):
        this_input_x = input_responses[index * self.batch_size: (index + 1) * self.batch_size]
        this_input_y = input_labels[index * self.batch_size: (index + 1) * self.batch_size]
        this_input_len = input_lengths[index * self.batch_size: (index + 1) * self.batch_size]
        return this_input_x, this_input_y, this_input_len

    def get_pred_batch(self, input_responses, input_lengths, index):
        this_input_x = input_responses[index * self.batch_size: (index + 1) * self.batch_size]
        this_input_len = input_lengths[index * self.batch_size: (index + 1) * self.batch_size]
        return this_input_x, this_input_len

# 是一个将回答的句子分成单词的函数
def read_emotional_response_label_file(train_res_file, train_label_file, max_len=30):
    f1 = open(train_res_file, "r", encoding="utf-8")
    f2 = open(train_label_file, "r", encoding="utf-8")
    res_lines = f1.readlines()
    label_lines = f2.readlines()


    train_responses = []
    train_labels = []
    train_lens = []
    for res_line, label_line in zip(res_lines, label_lines):
        # 去掉label首尾的空格
        label = label_line.strip()

        if label not in emotion_dict.keys():
            continue
        words = res_line.strip().split()
        print(words)
        # split()函数是用来分割句子的，用空格和/n来切分,其实就是把句子分成一个个单词
        if len(words) > max_len:
            # 如果超过了，就直接砍掉后面多余的
            words = words[: max_len]
        train_responses.append(words)
        # 在response后面把words拼上去，其实也就是把response分成单词放进train_responses里面
        train_labels.append(emotion_dict[label])
        # 把情绪的也放进去
        train_lens.append(len(words))
        # 把回答的长度也放进去
    return train_responses, train_labels, train_lens

# 是一个将回答的句子分成单词的函数
def read_test_emotional_response_label_file(train_res_file, max_len=30):
    f1 = open(train_res_file, "r", encoding="utf-8")
    res_lines = f1.readlines()
    print("lines from file")
    print(len(res_lines))
    train_responses = []
    train_lens = []
    for res_line in res_lines:
        words = res_line.strip().split()
        # print(words)
        # split()函数是用来分割句子的，用空格和/n来切分,其实就是把句子分成一个个单词
        if len(words) > max_len:
            # 如果超过了，就直接砍掉后面多余的
            words = words[: max_len]
        train_responses.append(words)
        # 把情绪的也放进去
        train_lens.append(len(words))
        # 把回答的长度也放进去
    return train_responses, train_lens

def read_test_data(filename, max_len):
    f = open(filename, "r", encoding="utf-8")
    lines = f.readlines()

    responses = []
    train_lens = []
    for line in lines:
        words = line.strip().split()
        if len(words) > max_len:
            words = words[: max_len]
        responses.append(words)
        train_lens.append(len(words))
    return responses, train_lens

# 补齐responses的函数（补齐成max_len）
def response_to_indexs(train_responses, word_dict, word_unk_id, max_len):
    new_responses = []
    for response in train_responses:
        # 在通用词词典里找到句子中的单词，如果没有就用unk
        new_response = [word_dict[word] if word in word_dict else word_unk_id for word in response]
        if len(new_response) < max_len:
            # 如果长度不够就用unk补齐
            remain = max_len - len(new_response)
            for i in range(remain):
                new_response.append(word_unk_id)
        new_responses.append(new_response)
    return new_responses

# 处理不能被batchsize整除的样本集
def align_train_batch_size(train_responses, emotion_labels, response_lens, batch_size):
    length = len(train_responses)
    if length % batch_size != 0:  # 如果不能被完整分批次
        remain = batch_size - length % batch_size  # 是去掉多余的后剩下有多少
        total_data = [[res, label, length] for res, label, length in
                      zip(train_responses, emotion_labels, response_lens)]
        sequence = range(length)
        for _ in range(remain):
            index = random.choice(sequence)  # 会随机选中一个response的index
            total_data.append(total_data[index])
        train_responses = [data[0] for data in total_data]
        emotion_labels = [data[1] for data in total_data]
        response_lens = [data[2] for data in total_data]
    return train_responses, emotion_labels, response_lens

def align_test_batch_size(test_response_without,response_lens, batch_size):
    length = len(test_response_without)
    if length % batch_size != 0:  # 如果不能被完整分批次
        remain = batch_size - length % batch_size  # 是去掉多余的后剩下有多少
        total_data = [[res, length] for res, length in
                      zip(test_response_without, response_lens)]
        sequence = range(length)
        for _ in range(remain):
            index = random.choice(sequence)  # 会随机选中一个response的index
            total_data.append(total_data[index])
        test_response_without = [data[0] for data in total_data]
        response_lens = [data[1] for data in total_data]
    return test_response_without, response_lens



def shuffle_train_data(train_responses, emotion_labels, response_lens):
    total_data = [[res, label, length] for res, label, length in
                  zip(train_responses, emotion_labels, response_lens)]
    # 随机排序
    random.shuffle(total_data)
    train_responses = [data[0] for data in total_data]
    emotion_labels = [data[1] for data in total_data]
    response_lens = [data[2] for data in total_data]
    return train_responses, emotion_labels, response_lens


def write_labels(file_name, labels):
    f = open(file_name, "w", encoding="utf-8")
    for label in labels:
        f.write(id2emotion[label])
        f.write("\n")
    f.close()


def split_train_valid_data(train_responses, train_labels, train_lens):
    # 回答的总长度，即有多少个回答
    total_length = len(train_responses)
    train_len = int(total_length * 0.9)
    # 取总长度的0.1
    valid_len = total_length - train_len
    # 随意取total_length里的一部分作为valid_len范围（验证集）里的有效长度
    sequence = random.sample(range(total_length), valid_len)
    training_res, training_labels, training_lens, valid_res, valid_labels, valid_lens = [], [], [], [], [], []
    for i in range(total_length):
        # 如果i在valid_len里面
        if i in sequence:
            valid_res.append(train_responses[i])  # 把对应的回答放进去
            valid_labels.append(train_labels[i])  # 把标签放进去
            valid_lens.append(train_lens[i])  # 把这个回答的长度放进去（选中的这个回答的长度）
        else:  # 如果不是，就不要放进valid数组
            training_res.append(train_responses[i])
            training_labels.append(train_labels[i])
            training_lens.append(train_lens[i])
    return training_res, training_labels, training_lens, valid_res, valid_labels, valid_lens


def compute_accuracy(pred_labels, true_labels):
    total_len = len(pred_labels)
    num = 0
    for pred, true_label in zip(pred_labels, true_labels):
        if pred == true_label:
            num += 1
    acc = float(num) / total_len
    return acc

# 读embeddings文件并且转成embedding的函数
def read_total_embeddings(embedding_file, vocab_size):
    embeddings = list()
    # 构造一个词典
    word2id = dict()
    id2word = dict()
    word_list=[]
    f = open(embedding_file, "r", encoding="utf-8")
    for line in f.readlines()[: vocab_size]:
        # strip() 方法用于移除字符串头尾指定的字符（默认为空格或换行符）或字符序列。
        # split()函数是用来分割句子的，用空格和/n来切分,其实就是把句子分成一个个单词
        lemmas = line.strip().split()
        # 说明这个文件每一行的第一个单词都是这个embedding代表的单词
        word = lemmas[0].strip()
        embedding = list()
        for lemma in lemmas[1:]:
            # 把str模式的float转成真正的float
            embedding.append(float(lemma.strip()))
        # print("embedding:" + str(len(embedding)))
        index = len(word2id)
        # 制作单词到index的词典
        word2id[word] = index
        # 制作index到单词的词典
        id2word[index] = word
        embeddings.append(embedding)
        word_list.append(word)

    return embeddings, word2id, id2word,word_list

def index_test_data(test_data, id2words):
    """
    write post and most correct response data
    :param test_data:
    :param write_file:
    :param id2words:
    :param test_emotion_labels:
    :return:
    """
    # f = open(write_file, "w", encoding="utf-8")
    generate_data=[]
    for data in test_data:
        generate_data_b=[]
        for data_b in data:
            words = [id2words[index] if index in range(len(id2words)) else 0 for index in data_b]

            sentence = " ".join(str(word) for word in words)
            final_sentence = sentence
            generate_data_b.append(final_sentence)
        generate_data.append(generate_data_b)
    return generate_data

def change_file_format(origin):
    # [origin]的形状是：[batch_size,beam_size,max_len]
    start_symbol = "<ss>"
    end_symbol = "<es>"
    new=[]

    # f1 = open(file1, "r", encoding="utf-8")
    # f2 = open(file2, "w", encoding="utf-8")
    for line in origin:
        new_b=[]
        for line_b in line:
            line_str=' '.join(line_b)
            words = line_str.strip().split()
            words = words[1:]
            if start_symbol in words:
                start_index = words.index(start_symbol) + 1
            else:
                start_index = 0
            if end_symbol in words:
                end_index = words.index(end_symbol)
            else:
                end_index = len(words)
            selected_words = words[start_index: end_index]
            sentence = " ".join(selected_words)
            new_b.append(sentence)
        new.append(new_b)
    return new

def add_emo_beam(beam_size,emos_data):
    # emos_data的形状是：[batch_size,1]
    new_emos=[]
    for emo in emos_data:
        new_emos_in = []
        for i in range(beam_size):
            new_emos_in.append(emo)
        new_emos.append(new_emos_in)
        # new_emos的形状就变成了[batch_size,beam_size]
        # 并且同一个beam里，情感都是一样的
    return new_emos
    # for i in range(beam_size):
    #     new_emos_in = []
    #     for emo in emos_data:
    #         new_emos_in.append(emo)
    #     new_emos.append(new_emos_in)
    #     # new_emos的形状就变成了[batch_size,beam_size]
    #     # 并且同一个beam里，情感都是一样的
    # return new_emos


def read_beam_test(test_res,test_emo,max_len=30):
    # test_res:[batch_size,beam,max_len]
    # test_emo:[batch_size,beam]
    test_responses=[]
    test_lens=[]
    test_emos=[]
    for res_b,emo_b in zip(test_res,test_emo):
        test_responses_b = []
        test_lens_b = []
        test_emos_b = []
        for res,emo in zip(res_b,emo_b):
            words= res.strip().split()
            emo_s=emo
            if emo_s not in range(7):
                continue
            if len(words) > max_len:
                # 如果超过了，就直接砍掉后面多余的
                words = words[: max_len]
            test_responses_b.append(words)
            test_lens_b.append(len(words))
            test_emos_b.append(emo_s)
        test_responses.append(test_responses_b)
        test_lens.append(test_lens_b)
        test_emos.append(test_emos_b)
    return test_responses,test_lens,test_emos

def response_to_indexs_b(train_responses, word_dict, word_unk_id, max_len):
    new_responses = []
    for response_o in train_responses:
        new_responses_b=[]
        for response in response_o:
            # 在通用词词典里找到句子中的单词，如果没有就用unk
            new_response = [word_dict[word] if word in word_dict else word_unk_id for word in response]
            if len(new_response) < max_len:
                 # 如果长度不够就用unk补齐
                remain = max_len - len(new_response)
                for i in range(remain):
                    new_response.append(word_unk_id)
            new_responses_b.append(new_response)
        new_responses.append(new_responses_b)
    return new_responses

def train_b(train_response_file, train_label_file, test_res_file, test_label_file, max_len, word_count_file, vocab_size,
          embedding_file, embedding_size, batch_size, num_epoch, hidden_size, emotion_class, pred_label_file, session,generate_words, scores,this_emotion_labels,lstm_emotion_machine):
    print("read word and embeddings\n")
    model_path = os.path.dirname(os.path.dirname(os.path.abspath("lstm_classifier.py")))
    data_dir = os.path.join(model_path, "data")
    embeddings, total_word_dict, id2word,word_l=read_total_embeddings(embedding_file, vocab_size)
    word_unk_id = total_word_dict['</s>']

    print("reading training data\n")
    # 将回答的句子分成单词
    train_responses, train_labels, train_lens = read_emotional_response_label_file(train_response_file,
                                                                                   train_label_file,
                                                                                   max_len)
    # 补齐responses（补齐成max_len）
    train_responses = response_to_indexs(train_responses, total_word_dict, word_unk_id, max_len)
    # train_responses, train_labels什么的都会更新，会把valid的放出去，也就是随机分出验证集
    train_responses, train_labels, train_lens, valid_res, valid_labels, valid_lens = \
        split_train_valid_data(train_responses, train_labels, train_lens)
    # 处理不能被batch_size整除的train样本集，随机挑一些直接砍掉
    train_responses, train_labels, train_lens = align_train_batch_size(train_responses, train_labels, train_lens,
                                                                       batch_size)
    # 处理不能被batch_size整除的valid样本集，随机挑一些直接砍掉
    valid_res, valid_labels, valid_lens = align_train_batch_size(valid_res, valid_labels, valid_lens, batch_size)

    # 读测试数据，和训练集一样的操作
    print("Read prediction data!\n")
    test_responses,test_lens = read_test_emotional_response_label_file(test_res_file,max_len)
    print(len(test_responses))

    test_responses = response_to_indexs(test_responses, total_word_dict, word_unk_id, max_len)

    test_length = len(test_responses)
    test_responses, test_lens = align_test_batch_size(test_responses,test_lens, batch_size)

    # 定义bi-lstm的模型
    print("Define model!\n")
    lstm_emotion_machine = LstmClassifier(embeddings, word_dict, embedding_size, hidden_size, emotion_class, batch_size,max_len, True, session, learning_rate=0.1)
    # 开始训练
    # 开始训练
    print("training\n")
    # 一共有多少批训练集

    train_batch = int(len(train_responses) / batch_size)
    # 一共有多少批验证集
    valid_batch = int(len(valid_res) / batch_size)
    valid_accs = []
    best_valid_acc = -1.0
    ckpt_path = os.path.join(os.path.join(model_path, "data_utils/check_path_lstm"), "check_path_lstm")
    # 开始训练
    for i in range(num_epoch):
        print("Now train epoch %d!\n" % (i + 1))
        # 首先随机排序，即打乱顺序
        train_responses, train_labels, train_lens = shuffle_train_data(train_responses, train_labels, train_lens)

        for j in range(train_batch):
            # 处理输入，处理成能直接喂进去的形式
            this_res, this_label, this_len = lstm_emotion_machine.get_train_batch(train_responses, train_labels,
                                                                                  train_lens, j)
            # 计算损失
            loss = lstm_emotion_machine.train_step(this_res, this_label, this_len)
            print("epoch=%d, batch=%d, loss=%f\n" % ((i + 1), (j + 1), loss))
        # 验证集验证开始
        labels = []
        for k in range(valid_batch):
            this_res, this_label, this_len = lstm_emotion_machine.get_train_batch(valid_res, valid_labels,
                                                                                  valid_lens, k)
            # 输出预测出来的labels,可能有多个？
            this_labels = lstm_emotion_machine.predict_step(this_res, this_len)
            labels.extend(this_labels)
        # 计算准确度
        accuracy = compute_accuracy(labels, valid_labels)
        print("epoch=%d, accuracy=%f\n" % ((i + 1), accuracy))
        valid_accs.append(accuracy)
    # 保存这次的结果
        if best_valid_acc < accuracy:
            best_valid_acc = accuracy
            lstm_emotion_machine.saver.save(lstm_emotion_machine.sess, ckpt_path, global_step=(i + 1) * train_batch)
            #lstm_emotion_machine.save_weights(FLAGS.checkpoint_path)
    # 取最好的和平均的准确率
    best_acc = np.max(valid_accs)
    ave_acc = np.average(valid_accs)
    print("best acc=%f, average acc=%f\n" % (best_acc, ave_acc))


    # last_ckpt = lstm_emotion_machine.saver.last_checkpoints
    # print("test here")
    # print(last_ckpt)
    # restore_path = lstm_emotion_machine.saver.last_checkpoints[0]


    checkpoint_path = os.path.join(os.path.join(model_path, "data_utils/check_path_lstm"), "check_path_lstm-11728")


    # lstm_emotion_machine.saver.restore(lstm_emotion_machine.sess, checkpoint_path)

    # 移动过来的
    generate_words=index_test_data(generate_words,id2word)
    # 去掉start和end_id
    generate_words=change_file_format(generate_words)
    # 把emo也变成beam个重复的数据形状
    this_emotion_labels=add_emo_beam(20,this_emotion_labels)

    test_responses,test_lens,this_emotion_labels= read_beam_test(generate_words,this_emotion_labels,FLAGS.max_len)
    test_responses = response_to_indexs_b(test_responses, total_word_dict, word_unk_id, FLAGS.max_len)

    total_labels = []
    # for k in range(pred_batches):
    # 处理输入
    # this_res, this_len = lstm_emotion_machine.get_pred_batch(test_responses, chat_config.max_len, k)
    # 输出预测的labels，形状是[batch_size,beam,1]
    # tf.reset_default_graph()
    this_labels_scores = lstm_emotion_machine.beam_predict_step(test_responses, test_lens, this_emotion_labels)
    scores_mul = []
    max_col = []
    for ge_score, emo_score in zip(scores, this_labels_scores):
        scores_mul_b = []
        for ge_score_b, emo_score_b in zip(ge_score, emo_score):
            scores_mul_b.append(ge_score_b * emo_score_b)
        scores_mul.append(scores_mul_b)
    # max_col的形状是[batch_size,1],类型是tensor
    for scores in scores_mul:
        max_index = scores.index(max(scores))
        max_col.append(max_index)
    for i in range(batch_size):
        print(max_col[i])

        total_labels.append(generate_words[i][max_col[i]])
    return total_labels

    '''
    pred_batches = int(len(test_responses) / batch_size)
    total_labels = []
    for k in range(pred_batches):
        # 处理输入
        this_res, this_len = lstm_emotion_machine.get_pred_batch(test_responses, test_lens, k)
        # 输出预测的labels
        this_labels = lstm_emotion_machine.predict_step(this_res, this_len)
        total_labels.extend(this_labels)

    print("check here length")
    print(len(test_responses))
    print(len(total_labels))
    write_labels(pred_label_file, total_labels)

    return lstm_emotion_machine
    '''
    # print(len(total_labels))


def train(config_file,train_response_file, train_label_file, test_res_file, test_label_file, max_len, word_count_file, vocab_size,
          embedding_file, embedding_size, batch_size, num_epoch, hidden_size, emotion_class, pred_label_file,pre_train_word_count_file,emotion_words_dir, session):
    chat_config = ChatConfig(config_file)
    print("read word and embeddings\n")
    total_embeddings, total_word2id, total_id2word,total_word_list,=read_total_embeddings(embedding_file, vocab_size)
    pre_word_count = get_word_count(pre_train_word_count_file, chat_config.word_count)
    emotion_words_dict = read_emotion_words(emotion_words_dir, pre_word_count)
    word_list = construct_vocab(total_word_list, emotion_words_dict, chat_config.generic_word_size,chat_config.emotion_vocab_size, FLAGS.unk)
    word_dict = construct_word_dict(word_list, FLAGS.unk, FLAGS.start_symbol, FLAGS.end_symbol)
    id2words = {idx: word for word, idx in word_dict.items()}
    word_unk_id = word_dict[FLAGS.unk]
    word_start_id = word_dict[FLAGS.start_symbol]
    word_end_id = word_dict[FLAGS.end_symbol]
    final_word_list = get_word_list(id2words)
    print("Read word embeddings!\n")
    # 读所有的词向量
    embeddings = read_word_embeddings(total_embeddings, total_word2id, final_word_list, chat_config.embedding_size)

    print("reading training data\n")
    # 将回答的句子分成单词
    train_responses, train_labels, train_lens = read_emotional_response_label_file(train_response_file,
                                                                                   train_label_file,
                                                                                   max_len)
    # 补齐responses（补齐成max_len）
    train_responses = response_to_indexs(train_responses, word_dict, word_unk_id, max_len)
    # train_responses, train_labels什么的都会更新，会把valid的放出去，也就是随机分出验证集
    train_responses, train_labels, train_lens, valid_res, valid_labels, valid_lens = \
        split_train_valid_data(train_responses, train_labels, train_lens)
    # 处理不能被batch_size整除的train样本集，随机挑一些直接砍掉
    train_responses, train_labels, train_lens = align_train_batch_size(train_responses, train_labels, train_lens,
                                                                       batch_size)

    # 处理不能被batch_size整除的valid样本集，随机挑一些直接砍掉
    valid_res, valid_labels, valid_lens = align_train_batch_size(valid_res, valid_labels, valid_lens, batch_size)

    # 读测试数据，和训练集一样的操作
    print("Read prediction data!\n")
    test_responses,test_lens = read_test_emotional_response_label_file(test_res_file,max_len)
    print(len(test_responses))

    test_responses = response_to_indexs(test_responses, word_dict, word_unk_id, max_len)

    test_length = len(test_responses)
    test_responses, test_lens = align_test_batch_size(test_responses,test_lens, batch_size)

    # 定义bi-lstm的模型
    print("Define model!\n")
    lstm_emotion_machine = LstmClassifier(embeddings, word_dict, embedding_size, hidden_size, emotion_class, batch_size,max_len, True, session, learning_rate=0.1)
    # 开始训练
    print("training\n")
    # 一共有多少批训练集
    train_batch = int(len(train_responses) / batch_size)
    # 一共有多少批验证集
    valid_batch = int(len(valid_res) / batch_size)
    valid_accs = []
    best_valid_acc = -1.0
    ckpt_path = os.path.join(FLAGS.checkpoint_path, "check_path_lstm")
    # 开始训练
    for i in range(num_epoch):
        print("Now train epoch %d!\n" % (i + 1))
        # 首先随机排序，即打乱顺序
        train_responses, train_labels, train_lens = shuffle_train_data(train_responses, train_labels, train_lens)

        for j in range(train_batch):
            # 处理输入，处理成能直接喂进去的形式
            this_res, this_label, this_len = lstm_emotion_machine.get_train_batch(train_responses, train_labels,
                                                                                  train_lens, j)
            # 计算损失
            loss = lstm_emotion_machine.train_step(this_res, this_label, this_len)
            print("epoch=%d, batch=%d, loss=%f\n" % ((i + 1), (j + 1), loss))
        # 验证集验证开始
        labels = []
        for k in range(valid_batch):
            this_res, this_label, this_len = lstm_emotion_machine.get_train_batch(valid_res, valid_labels,
                                                                                  valid_lens, k)
            # 输出预测出来的labels,可能有多个？
            this_labels = lstm_emotion_machine.predict_step(this_res, this_len)
            labels.extend(this_labels)
        # 计算准确度
        accuracy = compute_accuracy(labels, valid_labels)
        print("epoch=%d, accuracy=%f\n" % ((i + 1), accuracy))
        valid_accs.append(accuracy)
    # 保存这次的结果
        if best_valid_acc < accuracy:
            best_valid_acc = accuracy
            lstm_emotion_machine.saver.save(lstm_emotion_machine.sess, ckpt_path, global_step=(i + 1) * train_batch)
            #lstm_emotion_machine.save_weights(FLAGS.checkpoint_path)
    # 取最好的和平均的准确率
    best_acc = np.max(valid_accs)
    ave_acc = np.average(valid_accs)
    print("best acc=%f, average acc=%f\n" % (best_acc, ave_acc))

    #开始测试
    # restore_path = lstm_emotion_machine.saver.last_checkpoints[-1]
    # lstm_emotion_machine.saver.restore(lstm_emotion_machine.sess, restore_path)
    #
    #
    # restore_path = lstm_emotion_machine.saver.last_checkpoints[0]
    # checkpoint_path = os.path.join(os.path.join(model_path, "data_utils/check_path_lstm"), "check_path_lstm-11728")
    #
    # lstm_emotion_machine.saver.restore(lstm_emotion_machine.sess, checkpoint_path)
    pred_batches = int(len(test_responses) / batch_size)
    total_labels = []
    for k in range(pred_batches):
        # 处理输入
        this_res, this_len = lstm_emotion_machine.get_pred_batch(test_responses, test_lens, k)
        # 输出预测的labels
        this_labels = lstm_emotion_machine.predict_step(this_res, this_len)
        total_labels.extend(this_labels)

    print("check here length")
    print(len(test_responses))
    print(len(total_labels))
    last_ckpt = lstm_emotion_machine.saver.last_checkpoints
    print("test here")
    print(last_ckpt)
    write_labels(pred_label_file, total_labels)

    return lstm_emotion_machine
    # print(len(total_labels))




def main(_):
    with tf.device("/gpu:1"):
        sess = tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True, log_device_placement=True))
        train(FLAGS.config_file,FLAGS.train_response_file, FLAGS.train_label_file, FLAGS.test_response_file, FLAGS.test_label_file,
              FLAGS.max_len, FLAGS.word_count_file, FLAGS.vocab_size, FLAGS.embedding_file, FLAGS.embedding_size,
              FLAGS.batch_size, FLAGS.num_epoch, FLAGS.hidden_size, FLAGS.emotion_class, FLAGS.pred_label_file,FLAGS.pre_train_word_count_file,FLAGS.emotion_words_dir, sess)


if __name__ == "__main__":
    model_path = os.path.dirname(os.path.dirname(os.path.abspath("lstm_classifier.py")))
    data_dir = os.path.join(model_path, "data")

    parse = argparse.ArgumentParser()
    parse.add_argument("--train_response_file", type=str,
                       default=os.path.join(data_dir, "stc_data/test/utt123_trans_CN_jieba.txt"))
    parse.add_argument("--train_label_file", type=str,
                       default=os.path.join(data_dir, "stc_data/test/utt123_emo_CN.txt"))
    # parse.add_argument("--test_response_file", type=str,
    #                    default="/Users/zhouziyi/Desktop/Lab/Graduate Design/corpus/douban/post.without.txt")
    parse.add_argument("--test_response_file", type=str,
                       default=os.path.join(data_dir, "stc_data/test/trans/gen_ex_pre.final.txt"))
    # test label是用于计算测试准确度的，不涉及别的
    parse.add_argument("--test_label_file", type=str,
                       default=os.path.join(data_dir, "stc_data/train_test/test.label.lstm.filter.txt"))
    parse.add_argument("--max_len", type=int, default=15)
    # word_count_file并没有用处
    parse.add_argument("--word_count_file", type=str,
                       default=os.path.join(data_dir, "emotion_words_human/word.count.7.120.CN.txt"))
    parse.add_argument("--vocab_size", type=int, default=40000)
    parse.add_argument("--embedding_file", type=str,
                       default=os.path.join(data_dir, "embedding/7_classes_trans_metric.txt"),
                       help="word embedding file path")
    parse.add_argument("--embedding_size", type=int, default=50)
    parse.add_argument("--batch_size", type=int, default=8)
    parse.add_argument("--num_epoch", type=int, default=500)
    parse.add_argument("--hidden_size", type=int, default=256)
    parse.add_argument("--emotion_class", type=int, default=7)
    # pred_label_file并没有用处
    parse.add_argument("--pred_label_file", type=str, default=os.path.join(data_dir, "com_data/5-19-check/generated_sentences_2.final_lstm_classifier.txt"))
    parse.add_argument("--unk", type=str, default="</s>", help="symbol for unk words")
    parse.add_argument("--start_symbol", type=str, default="<ss>", help="symbol for response sentence start")
    parse.add_argument("--end_symbol", type=str, default="<es>", help="symbol for response sentence end")
    parse.add_argument("--learning_rate", type=float, default=0.1)
    parse.add_argument("--checkpoint_path", type=str, default=os.path.join(model_path, "data_utils/check_path_lstm_ori_ex"))
    # parse.add_argument("--checkpoint_path", type=str, default=os.path.join(model_path, "data_utils"))
    parse.add_argument("--pre_train_word_count_file", type=str,
                       default=os.path.join(data_dir, "emotion_words_human/word.count.7.120.CN.txt"),
                       help="nlp cc word count file")
    parse.add_argument("--emotion_words_dir", type=str,
                       default=os.path.join(data_dir, "emotion_words_human/7_class_CN_120"),
                       help="emotion words directory")
    parse.add_argument("--config_file", type=str, default=os.path.join(model_path, "conf/dialogue1.conf"),
                       help="configuration file path")

    FLAGS, unparsed = parse.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
















