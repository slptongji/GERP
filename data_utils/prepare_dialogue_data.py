import os
import random
import numpy as np
np.set_printoptions(threshold=np.inf)

__author__ = "Song"
# emotion_dict = {"anger": 0, "disgust": 1, "happiness": 2, "like": 3, "sadness": 4, "neutral": 5}
emotion_dict = {"anger": 0, "disgust": 1, "fear": 2, "joy": 3, "neutral": 4, "sadness": 5, "surprise": 6}
id2emotion = {idx: emotion for emotion, idx in emotion_dict.items()}


def get_word_count(word_count_file, min_word_count):
    f = open(word_count_file, "r", encoding="utf-8")

    word_dict = dict()
    for line in f.readlines():
        # lemmas是以|||分割出来的单词
        lemmas = line.strip().split("|||")
        word = lemmas[0].strip()
        count = int(lemmas[1].strip())
        word_dict[word] = count
        """
        if count >= min_word_count:
            word_dict[word] = count
        """
    f.close()
    return word_dict

# 是从分好类的情感文件中读词出来构造情感词典
def read_emotion_words(data_dir, word_dict):
    emotion_words_dict = dict()
    for i in range(7):
        # 分类的情感词典，每个情感内部建一个
        emotion_words_dict[id2emotion[i]] = dict()
        filename = "emotion.word." + id2emotion[i] + ".txt"
        file_path = os.path.join(data_dir, filename)
        f = open(file_path, "r", encoding="utf-8")
        for line in f.readlines():
            # word是不同情感分类中的情感单词，例如：sadness,anger这种
            word = line.strip()
            if word not in word_dict.keys():
                continue
            emotion_words_dict[id2emotion[i]][word] = word_dict[word]
        f.close()
        print("dict length")
        print(id2emotion[i])
        print(len(emotion_words_dict[id2emotion[i]]))
    return emotion_words_dict


def read_total_embeddings(embedding_file, max_vocab_size):
    embeddings = list()
    word2id = dict()
    word_list = list()
    f = open(embedding_file, "r", encoding="utf-8")
    for line in f.readlines()[: max_vocab_size]:
        lemmas = line.strip().split()
        word = lemmas[0].strip()
        word_list.append(word)
        embedding = list()
        for lemma in lemmas[1:]:
            embedding.append(float(lemma.strip()))
        index = len(word2id)
        word2id[word] = index
        embeddings.append(embedding)
    return embeddings, word2id, word_list


def construct_vocab(word_list, emotion_word_dict, generic_word_size, emotion_word_num, word_unk):
    """

    :param word_list:
    :param emotion_word_dict:
    :param generic_word_size:
    :param emotion_word_num:
    :param word_unk:
    :return: The first total num words are generic words, and the next per emotion num words represent emotion words
    """
    total_emotion_word = list()
    emotion_word_num = 113
    # 依次取出每个情感类别中的单词
    for i in range(7):
        emotion_word_lists = emotion_word_dict[id2emotion[i]]
        print("lengrh check")
        print(len(emotion_word_lists))
        sorted_emotion_words = sorted(emotion_word_lists.items(), key=lambda x: x[1], reverse=True)

        length = 0
        index = 0
        print('sorted lengthL')
        print(len(sorted_emotion_words))

        while index < emotion_word_num:

            temp_word = sorted_emotion_words[index][0]
            if temp_word not in total_emotion_word:
                total_emotion_word.append(temp_word)
                length += 1
            index += 1

    new_word_list = list()
    length = 0
    index = -1
    while index < generic_word_size:
        index += 1
        print("check list here")
        print(len(word_list))
        if word_list[index] != word_unk and word_list[index] not in total_emotion_word:
            new_word_list.append(word_list[index])
            length += 1

    new_word_list.extend(total_emotion_word)
    # new word list: pre total:generic: per emotion: one emotion class
    return new_word_list


def construct_word_dict(words_list, unk="</s>", start_symbol="<ss>", end_symbol="<es>"):
    word_dict = dict()
    word_dict[unk] = 0
    word_dict[start_symbol] = 1
    word_dict[end_symbol] = 2
    for word in words_list:
        word_dict[word] = len(word_dict)
    return word_dict


def get_word_list(id2words):
    word_list = []
    for i in range(len(id2words)):
        word_list.append(id2words[i])
    return word_list


def read_word_embeddings(total_embeddings, embedding_word2ids, word_list, embedding_size):
    filter_embeddings = list()
    for word in word_list:
        if word in embedding_word2ids:
            index = embedding_word2ids[word]
            embedding = total_embeddings[index]
            # print("yes embedding")
            # print(embedding)
        else:
            embedding = list()
            for i in range(0, embedding_size):
                embedding.append(float(0))
            #embedding_ = np.zeros([embedding_size], dtype=float)
            #embedding = list(embedding_)
        filter_embeddings.append(embedding)
    return filter_embeddings


def read_training_file(filename, word_dict, unk="<unk>"):
    unk_id = word_dict[unk]

    train_data = []
    f = open(filename, "r", encoding="utf-8")
    for line in f.readlines():
        words = line.strip().split()
        word_indexes = [word_dict[word] if word in word_dict else unk_id for word in words]
        train_data.append(word_indexes)
    return train_data


def is_valid_sen(data, min_len, max_len):
    if (len(data) > min_len) and (len(data) <= max_len):
        return True
    else:
        return False


def filter_sentence_length(post_data, response_data, emotion_labels, min_len, max_len):
    new_post_data, new_response_data, new_emotion_labels = [], [], []
    for post, response, label in zip(post_data, response_data, emotion_labels):
        if is_valid_sen(post, min_len, max_len) and is_valid_sen(response, min_len, max_len):
            new_post_data.append(post)
            new_response_data.append(response)
            new_emotion_labels.append(label)
    return new_post_data, new_response_data, new_emotion_labels

def filter_sentence_length_peld(utt1_data, utt2_data, emo1_data,emo2_data,emo3_data,person_data, min_len, max_len):
    new_utt1, new_utt2,new_utt3, new_emo1,new_emo2,new_emo3,new_person = [], [], [],[],[],[],[]
    for utt1, utt2, emo1,emo2,emo3,person in zip(utt1_data, utt2_data, emo1_data,emo2_data,emo3_data,person_data):
        if is_valid_sen(utt1, min_len, max_len) and is_valid_sen(utt2, min_len, max_len):
            new_utt1.append(utt1)
            new_utt2.append(utt2)
            new_emo1.append(emo1)
            new_emo2.append(emo2)
            new_emo3.append(emo3)
            new_person.append(person)
    return new_utt1, new_utt2, new_emo1,new_emo2,new_emo3,new_person


"""
def filter_sentence_length(post_data, response_data, emotion_labels, min_len, max_len):
    new_post_data, new_response_data, new_emotion_labels = [], [], []
    for post, response, label in zip(post_data, response_data, emotion_labels):
        if not is_valid_sen(post, min_len, max_len):
            post = post[: max_len]
        if not is_valid_sen(response, min_len, max_len):
            response = response[: max_len]
        new_post_data.append(post)
        new_response_data.append(response)
        new_emotion_labels.append(label)
    return new_post_data, new_response_data, new_emotion_labels
"""

def filter_test_peld_sentence_length(utt1_data, utt2_data,emo1_data,emo2_data,person_data, min_len, max_len):
    new_utt1, new_utt2,new_emo1,new_emo2,new_per = [], [],[],[],[]
    for utt1, utt2,emo1,emo2,person in zip(utt1_data, utt2_data,emo1_data,emo2_data,person_data):
        if is_valid_sen(utt1, min_len, max_len) and is_valid_sen(utt2,min_len,max_len):
            new_utt1.append(utt1)
            new_utt2.append(utt2)
            new_emo1.append(emo1)
            new_emo2.append(emo2)
            new_per.append(person)
    return new_utt1, new_utt2,new_emo1,new_emo2,new_per

def filter_test_sentence_length(post_data, emotion_labels, min_len, max_len):
    new_post_data, new_emotion_labels = [], []
    for post, label in zip(post_data, emotion_labels):
        if is_valid_sen(post, min_len, max_len):
            new_post_data.append(post)
            new_emotion_labels.append(label)
    return new_post_data, new_emotion_labels


def align_sentence_length(train_data, max_len, unk_id):
    new_train_data = list()
    for sen_words in train_data:
        if len(sen_words) >= max_len:
            sen_words = sen_words[: max_len]
        else:
            remain = max_len - len(sen_words)
            for i in range(remain):
                sen_words.append(unk_id)
        new_train_data.append(sen_words)
    return new_train_data

# response_data是一个list[list[]] 是输入的样本中的回答
def get_predict_train_response_data(response_data, start_id, end_id, unk_id, max_len):
    predict_response_data = list()
    train_response_data = list()
    for data in response_data:
        this_pre_data, this_train_data = [], []
        if len(data) >= max_len - 1:
            # 如果超过了最长长度，就截断
            # predata是尾巴上有end标识的，用于在损失函数中的计算
            this_pre_data.extend(data[: max_len - 1])
            this_pre_data.append(end_id)
            # train_data是头部为start标识的回答
            this_train_data.append(start_id)
            this_train_data.extend(data[: max_len - 1])
        else:
            remain = max_len - len(data) - 1
            this_pre_data.extend(data)
            this_pre_data.append(end_id)

            this_train_data.append(start_id)
            this_train_data.extend(data)
            for _ in range(remain):
                this_pre_data.append(unk_id)
                this_train_data.append(unk_id)
        predict_response_data.append(this_pre_data)
        train_response_data.append(this_train_data)
    return train_response_data, predict_response_data

# response_data是一个list[list[]] 是输入的样本中的回答
def get_train_response_data(response_data, start_id, end_id, unk_id, max_len):

    train_response_data = list()
    for data in response_data:
        this_train_data = []
        if len(data) >= max_len - 1:
            # 如果超过了最长长度，就截断
            # train_data是头部为start标识的回答
            this_train_data.append(start_id)
            this_train_data.extend(data[: max_len - 1])
        else:
            remain = max_len - len(data) - 1

            this_train_data.append(start_id)
            this_train_data.extend(data)
            for _ in range(remain):

                this_train_data.append(unk_id)
        train_response_data.append(this_train_data)
    return train_response_data

def read_emotion_label(emotion_label_file):
    emotion_labels = list()
    f = open(emotion_label_file, "r", encoding="utf-8")
    for line in f.readlines():
        label = line.strip()
        emotion_labels.append(emotion_dict[label])
    return emotion_labels

def read_personality_file(per_file):
    per_list=list()
    f=open(per_file,"r",encoding="utf-8")
    for per in f.readlines():
        pers_sen=per.strip()
        pers_list=pers_sen.split(',')
        pers_list_int=[int(per) for per in pers_list]
        per_list.append(pers_list_int)
    return read_personality_file()


def read_emotion_profile(emotion_profile_file):
    emotion_profile = dict()
    f = open(emotion_profile_file,"r",encoding="utf-8")
    for line in f.readlines():
        words = line.strip().split()
        emotion_key = emotion_dict[words[0]]
        emotion_value = emotion_dict[words[1]]
        # 转成数字表示
        emotion_profile[emotion_key] = emotion_value

    return emotion_profile

def read_test_res_label(emotion_profile_dict,test_post_label,test_label_file):
    test_res_label = list()
    f = open(test_label_file,"w",encoding="utf-8")
    for label in test_post_label:
        if label in emotion_profile_dict:
            res_label = emotion_profile_dict[label]
            test_res_label.append(res_label)
            f.write(id2emotion[res_label])
            f.write("\n")
    return test_res_label


def align_batch_size(post_data, post_data_len, train_response_data, predict_response_data, emotion_labels, batch_size):
    length = len(post_data)
    if length % batch_size != 0:
        remain = batch_size - length % batch_size
        total_data = [[post, post_len, train_res, predict_res, label] for post, post_len, train_res, predict_res, label
                      in zip(post_data, post_data_len, train_response_data, predict_response_data, emotion_labels)]
        sequence = range(length)
        for _ in range(remain):
            index = random.choice(sequence)
            total_data.append(total_data[index])
        post_data = [data[0] for data in total_data]
        post_data_len = [data[1] for data in total_data]
        train_response_data = [data[2] for data in total_data]
        predict_response_data = [data[3] for data in total_data]
        emotion_labels = [data[4] for data in total_data]
    return post_data, post_data_len, train_response_data, predict_response_data, emotion_labels

def align_peld_batch_size(train_utt1,train_utt1_length, train_utt2,train_utt2_length, train_emo1,train_emo2,train_emo3,train_per, batch_size):
    length = len(train_utt1)
    if length % batch_size != 0:
        remain = batch_size - length % batch_size
        total_data = [[utt1, utt1_len, utt2, utt2_len,emo1,emo2,emo3,per] for utt1, utt1_len, utt2,utt2_len, emo1,emo2,emo3,per
                      in zip(train_utt1,train_utt1_length, train_utt2,train_utt2_length, train_emo1,train_emo2,train_emo3,train_per)]
        sequence = range(length)
        for _ in range(remain):
            index = random.choice(sequence)
            total_data.append(total_data[index])
        train_utt1 = [data[0] for data in total_data]
        train_utt1_length = [data[1] for data in total_data]
        train_utt2 = [data[2] for data in total_data]
        train_utt2_length = [data[3] for data in total_data]
        train_emo1 = [data[4] for data in total_data]
        train_emo2 = [data[5] for data in total_data]
        train_emo3 = [data[6] for data in total_data]
        train_per = [data[7] for data in total_data]
    return train_utt1, train_utt1_length, train_utt2,train_utt2_length,train_emo1, train_emo2, train_emo3,train_per


def align_test_batch_size(post_data, post_length, emotion_labels, batch_size):
    length = len(post_data)
    if length % batch_size != 0:
        # print("Length not enough!")
        remain = batch_size - length % batch_size
        total_data = [[post, length, label] for post, length, label in zip(post_data, post_length, emotion_labels)]
        sequence = range(length)
        for _ in range(remain):
            index = random.choice(sequence)
            total_data.append(total_data[index])
        post_data = [data[0] for data in total_data]
        post_length = [data[1] for data in total_data]
        emotion_labels = [data[2] for data in total_data]
    return post_data, post_length, emotion_labels

def align_test_peld_batch_size(test_utt1,test_utt1_length, test_utt2,test_utt2_length, test_emo1,test_emo2,test_emo3,test_per, batch_size):
    length = len(test_utt1)
    if length % batch_size != 0:
        # print("Length not enough!")
        remain = batch_size - length % batch_size
        total_data = [[utt1, utt1_len, utt2, utt2_len,emo1,emo2,emo3,per] for utt1, utt1_len, utt2, utt2_len,emo1,emo2,emo3,per in zip(test_utt1,test_utt1_length, test_utt2,test_utt2_length, test_emo1,test_emo2,test_emo3,test_per)]
        sequence = range(length)
        for _ in range(remain):
            index = random.choice(sequence)
            total_data.append(total_data[index])
        test_utt1 = [data[0] for data in total_data]
        test_utt1_length = [data[1] for data in total_data]
        test_utt2 = [data[2] for data in total_data]
        test_utt2_length = [data[3] for data in total_data]
        test_emo1 = [data[4] for data in total_data]
        test_emo2 = [data[5] for data in total_data]
        test_emo3 = [data[6] for data in total_data]
        test_per = [data[7] for data in total_data]
    return test_utt1,test_utt1_length, test_utt2,test_utt2_length, test_emo1,test_emo2,test_emo3,test_per

def shuffle_train_data(post_data, post_data_length, train_response_data, predict_response_data, emotion_labels):
    total_data = [[post, post_len, train_res, predict_res, label] for post, post_len, train_res, predict_res, label in
                  zip(post_data, post_data_length, train_response_data, predict_response_data, emotion_labels)]
    random.shuffle(total_data)
    post_data = [data[0] for data in total_data]
    post_data_length = [data[1] for data in total_data]
    train_response_data = [data[2] for data in total_data]
    predict_response_data = [data[3] for data in total_data]
    emotion_labels = [data[4] for data in total_data]
    return post_data, post_data_length, train_response_data, predict_response_data, emotion_labels


def sample_test_data(post_data, emotion_labels, train_post_length, sample_num):
    sequence = range(len(post_data))
    samples = random.sample(sequence, sample_num)
    sample_post, sample_emotion_labels, sample_post_lens = [], [], []
    for index in samples:
        sample_post.append(post_data[index])
        sample_emotion_labels.append(emotion_labels[index])
        sample_post_lens.append(train_post_length[index])
    return sample_post, sample_emotion_labels, sample_post_lens


def write_test_data(post_data,test_data, write_file, id2words, test_emotion_labels):
    """
    write post and most correct response data
    :param test_data:
    :param write_file:
    :param id2words:
    :param test_emotion_labels:
    :return:
    """
    f = open(write_file, "w", encoding="utf-8")
    for post,data, label in zip(post_data,test_data, test_emotion_labels):
        posts = [id2words[index] for index in post]
        sentence_p = " ".join(posts)
        words = [id2words[index] for index in data]
        sentence = " ".join(words)
        final_sentence = "post:"+ sentence_p+"  "+";"+id2emotion[label] + ": " + sentence
        f.write(final_sentence)
        print(final_sentence)
        f.write("\n")
    f.close()

def write_test_data_beam(post_data,test_data, write_file, id2words, test_emotion_labels):
    """
    write post and most correct response data
    :param test_data:
    :param write_file:
    :param id2words:
    :param test_emotion_labels:
    :return:
    """
    f = open(write_file, "w", encoding="utf-8")

    for post,data, label in zip(post_data,test_data, test_emotion_labels):
        posts = [id2words[index] for index in post]
        sentence_p = " ".join(posts)
        final_sentence = "post:"+ sentence_p+"  "+";"+id2emotion[label] + ": " + "<ss> " + data
        f.write(final_sentence)
        print(final_sentence)
        f.write("\n")
    f.close()


def write_generation_data(test_data, write_file, id2words, test_emotion_labels, batch_size, beam_size, max_len):
    """
    write all beam search data
    :param test_data: [len, batch, beam]
    :param write_file:
    :param id2words:
    :param test_emotion_labels:
    :param batch_size:
    :param beam_size:
    :param max_len:
    :return:
    """
    f = open(write_file, "w", encoding="utf-8")
    for i in range(batch_size):
        label = id2emotion[test_emotion_labels[i]]
        f.write(label)
        f.write("\n")
        print(label)
        for j in range(beam_size):
            words = []
            for k in range(max_len):
                words.append(id2words[test_data[k][i][j]])
            sentence = " ".join(words)
            f.write(sentence + "\n")
            print(sentence)
        f.write("\n")
        print("\n")
    f.close()


def read_stop_words(filename):
    stop_words = []
    f = open(filename, "r", encoding="utf-8")
    for line in f.readlines():
        stop_words.append(line.strip())
    return stop_words

























