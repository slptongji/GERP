import os
import sys
import random
import argparse

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.python.tools import inspect_checkpoint as chkp
import numpy as np
from src.configuration import ChatConfig
from data_utils.lstm_classifier import LstmClassifier,response_to_indexs,train_b,read_total_embeddings,read_emotional_response_label_file,split_train_valid_data,align_train_batch_size,shuffle_train_data,compute_accuracy
from data_utils.prepare_dialogue_data import construct_word_dict



emotion_dict = {"anger": 0, "disgust": 1, "fear": 2, "joy": 3, "neutral": 4, "sadness": 5, "surprise": 6}
id2emotion = {idx: emotion for emotion, idx in emotion_dict.items()}
FLAGS = None

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


def read_test_emotional_response_label(train_res, max_len=30):
    # f1 = open(train_res_file, "r", encoding="utf-8")
    res_lines = train_res
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
                print("cut here")
                continue
            if len(words) > max_len:
                # 如果超过了，就直接砍掉后面多余的
                words = words[: max_len]
            test_responses_b.append(words)
            print("length length")
            print(len(test_responses_b))
            test_lens_b.append(len(words))
            test_emos_b.append(emo_s)
        test_responses.append(test_responses_b)
        test_lens.append(test_lens_b)
        test_emos.append(test_emos_b)
    return test_responses,test_lens,test_emos




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
            # words =[id2words[index] for index in data_b]

            sentence = " ".join(str(word) for word in words)
            final_sentence = sentence

            generate_data_b.append(final_sentence)
        generate_data.append(generate_data_b)
    return generate_data




def select_best_response(generate_words, scores, this_post_data, this_emotion_labels,emotion_words_dict, batch_size, stop_words,word_start_id,word_end_id,word_unk_id,lstm_classi,emotion_chat_machine):
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

    def add_emo_beam(beam_size,emos_data):
        # emos_data的形状是：[batch_size,1]
        new_emos=[]
        for i in range(beam_size):
            new_emos_in = []
            for emo in emos_data:
                new_emos_in.append(emo)
            new_emos.append(new_emos_in)
        # new_emos的形状就变成了[batch_size,beam_size]
        # 并且同一个beam里，情感都是一样的
        return new_emos

    model_path = os.path.dirname(os.path.dirname(os.path.abspath("lstm_classifier.py")))
    data_dir = os.path.join(model_path, "data")
    parse = argparse.ArgumentParser()
    parse.add_argument("--start_symbol", type=str, default="<ss>", help="symbol for response sentence start")
    parse.add_argument("--end_symbol", type=str, default="<es>", help="symbol for response sentence end")
    parse.add_argument("--train_response_file", type=str,
                       default=os.path.join(data_dir, "stc_data/train/lstm_tran/utt1_trans_CN_jieba.txt"))
    parse.add_argument("--train_label_file", type=str,
                       default=os.path.join(data_dir, "stc_data/train/lstm_tran/utt1_trans_emo_CN.txt"))
    # parse.add_argument("--test_response_file", type=str,
    #                    default="/Users/zhouziyi/Desktop/Lab/Graduate Design/corpus/douban/post.without.txt")
    parse.add_argument("--test_response_file", type=str,
                       default=os.path.join(data_dir, "stc_data/test/trans/utt3_trans_generate_0.1_noLTS_8_noemo.final.txt"))
    # test label是用于计算测试准确度的，不涉及别的
    parse.add_argument("--test_label_file", type=str,
                       default=os.path.join(data_dir, "stc_data/train_test/test.label.lstm.filter.txt"))
    parse.add_argument("--max_len", type=int, default=15)
    # word_count_file并没有用处
    parse.add_argument("--word_count_file", type=str,
                       default=os.path.join(data_dir, "emotion_words_human/word.count.txt"))
    parse.add_argument("--vocab_size", type=int, default=40000)
    parse.add_argument("--embedding_file", type=str,
                       default=os.path.join(data_dir, "embedding/7_classes_trans_metric.txt"),
                       help="word embedding file path")
    parse.add_argument("--embedding_size", type=int, default=20)
    parse.add_argument("--batch_size", type=int, default=8)
    parse.add_argument("--num_epoch", type=int, default=50)
    parse.add_argument("--hidden_size", type=int, default=256)
    parse.add_argument("--emotion_class", type=int, default=7)
    # pred_label_file并没有用处
    parse.add_argument("--pred_label_file", type=str, default=os.path.join(data_dir, "stc_data/test/trans/lstm/utt3_trans_generate_0.1_noLTS_8.final_lstm_classifier.txt"))
    parse.add_argument("--unk", type=str, default='</s>', help="symbol for unk words")
    parse.add_argument("--learning_rate", type=float, default=0.1)
    parse.add_argument("--checkpoint_path", type=str, default=os.path.join(model_path, "data_utils/check_path_lstm_beam"))
    # parse.add_argument("--checkpoint_path", type=str, default=os.path.join(model_path, "data_utils"))

    FLAGS, unparsed = parse.parse_known_args()
    # chat_config = ChatConfig(os.path.join(model_path, "conf/dialogue1.conf"))

    # tf.reset_default_graph()
    # model_pred=LstmClassifier()
    # sess=tf.Session(config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=True))

    # saver=tf.train.import_meta_graph('check_path/lstm-classifier-119034.meta')
    # saver.restore(sess,tf.train.latest_checkpoint('check_path/'))
    # restore_path = tf.train.Saver().last_checkpoints[-1]
    # tf.train.Saver().restore(sess, restore_path)

    # f = open('log.beam.n.2.log', 'w')
    # sys.stdout = f
    # #
    # sys.stderr = f
    # tf.reset_default_graph()
    # reuse = tf.AUTO_REUSE
    # sess = sess_lstm

    embeddings, word_dict, id2word,word_list = read_total_embeddings(FLAGS.embedding_file, FLAGS.vocab_size)
    print(len(id2word))
    word_unk_id = word_dict[FLAGS.unk]
    word_dict = construct_word_dict(word_list, FLAGS.unk, FLAGS.start_symbol, FLAGS.end_symbol)
    id2word = {idx: word for word, idx in word_dict.items()}
    print("reading training data\n")
    # 将回答的句子分成单词
    train_responses, train_labels, train_lens = read_emotional_response_label_file(FLAGS.train_response_file,
                                                                                   FLAGS.train_label_file,
                                                                                   FLAGS.max_len)
    # 补齐responses（补齐成max_len）
    train_responses = response_to_indexs(train_responses, word_dict, word_unk_id, FLAGS.max_len)
    # train_responses, train_labels什么的都会更新，会把valid的放出去，也就是随机分出验证集
    train_responses, train_labels, train_lens, valid_res, valid_labels, valid_lens = \
        split_train_valid_data(train_responses, train_labels, train_lens)
    # 处理不能被batch_size整除的train样本集，随机挑一些直接砍掉
    train_responses, train_labels, train_lens = align_train_batch_size(train_responses, train_labels, train_lens,
                                                                       batch_size)
    # 处理不能被batch_size整除的valid样本集，随机挑一些直接砍掉
    valid_res, valid_labels, valid_lens = align_train_batch_size(valid_res, valid_labels, valid_lens, batch_size)
    # tf.reset_default_graph()
    '''
    lstm_emotion_machine = LstmClassifier(embeddings, word_dict, FLAGS.embedding_size, FLAGS.hidden_size, FLAGS.emotion_class, batch_size,
                                          FLAGS.max_len, True, session, learning_rate=FLAGS.learning_rate)
                                          '''
    train_batch = int(len(train_responses) / batch_size)
    # 一共有多少批验证集
    valid_batch = int(len(valid_res) / batch_size)
    valid_accs = []
    best_valid_acc = -1.0
    ckpt_path = os.path.join(FLAGS.checkpoint_path, "check_path_lstm")
    # res=train_b(FLAGS.train_response_file, FLAGS.train_label_file, FLAGS.test_response_file, FLAGS.test_label_file,
    #           FLAGS.max_len, FLAGS.word_count_file, FLAGS.vocab_size, FLAGS.embedding_file, FLAGS.embedding_size,
    #           FLAGS.batch_size, FLAGS.num_epoch, FLAGS.hidden_size, FLAGS.emotion_class, FLAGS.pred_label_file, sess, generate_words, scores, this_emotion_labels,lstm_emotion_machine)
    # return res
    # 开始训练
    '''
    for i in range(FLAGS.num_epoch):
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
    
    '''

    generate_words=index_test_data(generate_words,id2word)
    # 去掉start和end_id
    generate_words=change_file_format(generate_words)
    # 把emo也变成beam个重复的数据形状
    this_emotion_labels=add_emo_beam(FLAGS.max_len,this_emotion_labels)

    test_responses,test_lens,this_emotion_labels= read_beam_test(generate_words,this_emotion_labels,FLAGS.max_len)
    test_responses = response_to_indexs_b(test_responses, word_dict, word_unk_id, FLAGS.max_len)
    # test_responses, test_lens = align_test_batch_size(test_responses, test_lens, batch_size)

    # lstm_emotion_machine = LstmClassifier(embeddings, word_dict, 50, 256, 7, 8,30, True, session)
    # init = tf.global_variables_initializer()
    # lstm_emotion_machine.sess.run(init)
    # last_ckpt=lstm_classi.saver.last_checkpoints
    # print("test here")
    # print(last_ckpt)
    # restore_path = lstm_emotion_machine.saver.last_checkpoints[0]
    checkpoint_path = os.path.join(os.path.join(model_path, "data_utils/check_path_lstm_beam"), "check_path_lstm-733")
    # reader=pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
    #chkp.print_tensors_in_checkpoint_file(checkpoint_path,tensor_name=None,all_tensors=True)
    # lstm_emotion_machine=LstmClassifier.train()
    #lstm_emotion_machine.saver.restore(sess, checkpoint_path)
    print("stop here")
    # pred_batches = int(len(test_responses) / batch_size)
    total_labels = []
    # for k in range(pred_batches):
    # 处理输入
    # this_res, this_len = lstm_emotion_machine.get_pred_ba  tch(test_responses, chat_config.max_len, k)
    # 输出预测的labels，形状是[batch_size,beam,1]
    # tf.reset_default_graph()
    this_labels_scores = lstm_classi.beam_predict_step(test_responses, test_lens,this_emotion_labels)
    scores_mul=[]
    max_col=[]
    for ge_score, emo_score in zip(scores,this_labels_scores):
        scores_mul_b=[]
        for ge_score_b, emo_score_b in zip(ge_score,emo_score):
            scores_mul_b.append(ge_score_b*emo_score_b)
        scores_mul.append(scores_mul_b)
    # max_col的形状是[batch_size,1],类型是tensor
    for scores in scores_mul:
        max_index=scores.index(max(scores))
        max_col.append(max_index)
    # max_col=tf.argmax(scores_mul, axis=1)
   #  print("max_col here:")
   #  print(max_col)
   # # ax_col= lstm_emotion_machine.sess.run(max_col)
   #  max_col_vari=tf.Variable(max_col, name='max_col')
   #  init_new_vars_op = tf.variables_initializer([max_col_vari])
   #  sess=tf.Session()
   #  sess.run(init_new_vars_op)
   #  max_col_ten=tf.convert_to_tensor()
   #  # sess.run(tf.global_variables_initializer())
   #  max_col= sess.run(max_col)
   #  print(max_col)
    for i in range(batch_size):
        print(max_col[i])
        # print(max_col[i].item())
        # index=tf.cast(max_col[i],dtype=tf.int32,name=None)
        # print(index)
        total_labels.append(generate_words[i][max_col[i]])
    return total_labels