import tensorflow.compat.v1 as tf

from src.model import EmotionChatMachine
import random
from data_utils.lstm_classifier import LstmClassifier, add_emo_beam, response_to_indexs_b
from data_utils.beam import select_best_response, index_test_data, read_beam_test

tf.disable_v2_behavior()

def restore_Emo(checkpoint,config_file,word_dict,embeddings,chat_config,word_start_id,word_end_id,test_post_data,test_post_data_length,test_label_data):
    graph=tf.Graph()
    with tf.Session(graph=graph) as sess:
        emotion_chat_machine = EmotionChatMachine(config_file, sess, word_dict, embeddings,
                                              chat_config.generic_word_size + 3, word_start_id, word_end_id,
                                              "emotion_chat_machine")
        emotion_chat_machine.saver.restore(sess,checkpoint)
        test_batch = int(len(test_post_data) / chat_config.batch_size)
        print("test_batch")
        print(test_batch)
        generate_data = []
        scores_data=[]
        this_post_data_l,this_emotion_labels_l=[],[]
        for k in range(test_batch):
            this_post_data, this_post_len, this_emotion_labels, this_emotion_mask = \
                emotion_chat_machine.get_test_batch(test_post_data, test_post_data_length, test_label_data, k)
            generate_words, scores,embeddings = emotion_chat_machine.generate_step(this_post_data, this_post_len,
                                                                                        this_emotion_labels,
                                                                                        this_emotion_mask)
            generate_data.append(generate_words)
            scores_data.append(scores)
            print("this_emotion_labels.shape")

            print(len(this_emotion_labels))

            this_post_data_l.append(this_post_data)
            this_emotion_labels_l.append(this_emotion_labels)
        print("generate_data length")
        print(len(scores_data))
        return generate_data,scores_data,this_post_data_l,this_emotion_labels_l

def retore_LSTM(chat_config,checkpoint_path,generate_words,scores,this_post_data,this_emotion_labels,batch_size, word_unk_id,embeddings,word_dict,id2words):
    print("length of embedding")
    print(len(embeddings))
    graph_other=tf.Graph()
    with tf.Session(graph=graph_other) as sess_other:
        lstm_emotion_machine = LstmClassifier(embeddings, word_dict, 50, 256, 7, 8,15, True, sess_other)
        lstm_emotion_machine.saver.restore(sess_other, checkpoint_path)
        best_responses=[]
        print("check again")
        print(len(scores))

        for generate_words_b,scores_b,this_post_data_b,this_emotion_labels_b in zip(generate_words,scores,this_post_data,this_emotion_labels):
            generate_words_b = index_test_data(generate_words_b, id2words)
            print("Check~~~")
            print(len(generate_words_b))

            generate_words_b = change_file_format(generate_words_b)

            this_emotion_labels_b = add_emo_beam(chat_config.beam_size, this_emotion_labels_b)

            print("this_emotion_labels_b l")
            print(len(this_emotion_labels_b))
            test_responses_b, test_lens_b, this_emotion_labels_b = read_beam_test(generate_words_b, this_emotion_labels_b,
                                                                            chat_config.max_len)
            for b in test_responses_b:
                print("in test_responses_b length")
                print(len(b))
            print("test_responses_b")
            print(len(test_responses_b))
            test_responses_b = response_to_indexs_b(test_responses_b, word_dict, word_unk_id, chat_config.max_len)
            # for b in test_responses_b:
            #     print("in test_responses_b length")
            #     print(len(b))
            this_labels_scores = lstm_emotion_machine.beam_predict_step(test_responses_b, test_lens_b, this_emotion_labels_b)

            scores_mul_b = []
            max_col_b = []
            total_labels_b = []
            for ge_score, emo_score in zip(scores_b, this_labels_scores):
                # emo_score_b_max=np.max(emo_score,axis=1)
                scores_mul_i=[]
                for ge,emo in zip(ge_score,emo_score):
                    scores_mul_bl=0.8*ge + 0.2*emo
                    scores_mul_i.append(scores_mul_bl)
                scores_mul_b.append(scores_mul_i)
            # scores_mul_b[batch,beam_size]
            for scores_ in scores_mul_b:
                max_index = scores_.index(max(scores_))
                max_col_b.append(max_index)
            for i in range(batch_size):
                print(max_col_b[i])
                total_labels_b.extend(generate_words_b[i][max_col_b[i]])
            best_responses.extend(total_labels_b)
        return best_responses

def retore_LSTM_random(chat_config,checkpoint_path,generate_words,scores,this_post_data,this_emotion_labels,batch_size, word_unk_id,embeddings,word_dict,id2words):
    print("length of embedding")
    print(len(embeddings))
    graph_other=tf.Graph()
    with tf.Session(graph=graph_other) as sess_other:
        lstm_emotion_machine = LstmClassifier(embeddings, word_dict, 50, 256, 7, 8,15, True, sess_other)
        lstm_emotion_machine.saver.restore(sess_other, checkpoint_path)
        random_responses=[]
        print("check again")
        print(len(scores))

        for generate_words_b,scores_b,this_post_data_b,this_emotion_labels_b in zip(generate_words,scores,this_post_data,this_emotion_labels):
            generate_words_b = index_test_data(generate_words_b, id2words)

            generate_words_b = change_file_format(generate_words_b)

            this_emotion_labels_b = add_emo_beam(chat_config.beam_size, this_emotion_labels_b)

            print("this_emotion_labels_b l")
            print(len(this_emotion_labels_b))
            test_responses_b, test_lens_b, this_emotion_labels_b = read_beam_test(generate_words_b, this_emotion_labels_b,
                                                                            chat_config.max_len)
            for b in test_responses_b:
                print("in test_responses_b length")
                print(len(b))
            print("test_responses_b")
            print(len(test_responses_b))
            test_responses_b = response_to_indexs_b(test_responses_b, word_dict, word_unk_id, chat_config.max_len)
            # for b in test_responses_b:
            #     print("in test_responses_b length")
            #     print(len(b))
            this_labels_scores = lstm_emotion_machine.beam_predict_step(test_responses_b, test_lens_b, this_emotion_labels_b)

            scores_mul_b = []
            random_col_b = []
            total_labels_b = []
            for ge_score, emo_score in zip(scores_b, this_labels_scores):
                # emo_score_b_max=np.max(emo_score,axis=1)
                scores_mul_i=[]
                for ge,emo in zip(ge_score,emo_score):
                    scores_mul_bl=ge * emo
                    scores_mul_i.append(scores_mul_bl)
                scores_mul_b.append(scores_mul_i)
            # scores_mul_b[batch,beam_size]
            for scores_ in scores_mul_b:
                random_index= random.randint(0,len(scores_)-1)
                random_col_b.append(random_index)
            for i in range(batch_size):
                total_labels_b.append(generate_words_b[i][random_col_b[i]])
            random_responses.extend(total_labels_b)
        return random_responses

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
            sentence = " ".join('%s' %a for a in selected_words)
            new_b.append(sentence)
        new.append(new_b)
    return new
