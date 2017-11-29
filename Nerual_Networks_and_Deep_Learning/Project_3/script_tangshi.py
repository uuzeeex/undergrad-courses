# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 16:06:36 2017

@author: lamwa
"""

from rnn import *
import tensorflow as tf
import numpy as np

start_token = 'G'
end_token = 'E'

batch_size = 32

def run_training_tangshi():
    # 处理数据集
    _, word_to_int, vocabularies = process_poems('./poems.txt')
    poems_vector, _ = process_tangshi('./tangshi.txt', word_to_int)
    # 生成batch
    batches_inputs, batches_outputs = generate_batch(32, poems_vector, word_to_int)

    input_data = tf.placeholder(tf.int32, [batch_size, None])
    output_targets = tf.placeholder(tf.int32, [batch_size, None])
    # 构建模型
    end_points = rnn_model(model='lstm', input_data=input_data, output_data=output_targets, vocab_size=len(
        vocabularies), rnn_size=128, num_layers=2, batch_size=32, learning_rate=0.01)

    saver = tf.train.Saver()
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    with tf.Session() as sess:
        sess.run(init_op)
        saver.restore(sess, './poem_generator')
        for epoch in range(50):
            n = 0
            n_chunk = len(poems_vector) // batch_size
            for batch in range(n_chunk):
                loss, _, _ = sess.run([
                    end_points['total_loss'],
                    end_points['last_state'],
                    end_points['train_op']
                ], feed_dict={input_data: batches_inputs[n], output_targets: batches_outputs[n]})
                n += 1
                print('[INFO] Epoch: %d , batch: %d , training loss: %.6f' % (epoch, batch, loss))
        saver.save(sess, './poem_tangshi_generator')

def process_tangshi(file_name, word2index):
    poems = []
    with open(file_name, "r", encoding='utf-8', ) as f:
        poem = ''
        lines = f.readlines()
        for line in lines:
            if line.isspace() or line is lines[-1]:
                content = poem.strip()
                valid = True
                if '_' in content or '(' in content or '（' in content or '《' in content or '[' in content or \
                                start_token in content or end_token in content:
                    valid = False
                if len(content) < 5 or len(content) > 80:
                    valid = False
                if valid:
                    content = start_token + content + end_token
                    poems.append(content)
                poem = ''
            else:
                poem += line.strip()
    # 按诗的字数排序
    poems = sorted(poems, key=lambda line: len(line))
    # 统计每个字出现次数
    all_words = []
    for poem in poems:
        all_words += [word for word in poem]  
    counter = collections.Counter(all_words)  # 统计词和词频。
    count_pairs = sorted(counter.items(), key=lambda x: -x[1])  # 排序
    words, _ = zip(*count_pairs)
    words = words[:len(words)] + (' ',)
    word_int_map = dict(zip(words, range(len(words))))
    poems_vector = [list(map(word2index.get, poem)) for poem in poems]
    poems_vector = [poem_vector for poem_vector in poems_vector if not (None in poem_vector)]
    return poems_vector, words
    #return poems, lines

def gen_tangshi(begin_word):
    batch_size = 1
    _, word_int_map, vocabularies = process_poems('./poems.txt')
    poems_vector, _ = process_tangshi('./tangshi.txt', word_int_map)

    input_data = tf.placeholder(tf.int32, [batch_size, None])

    end_points = rnn_model(model='lstm', input_data=input_data, output_data=None, vocab_size=len(
        vocabularies), rnn_size=128, num_layers=2, batch_size=32, learning_rate=0.01)
    # 如果指定开始的字
    if begin_word:
        word = begin_word
    else:
        word = to_word(predict, vocabularies)
        
    saver = tf.train.Saver(tf.global_variables())
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    with tf.Session() as sess:
        sess.run(init_op)
        saver.restore(sess, './poem_tangshi_generator')# 恢复之前训练好的模型 
        poem = ''
        #???????????????????????????????????????
        # 下面部分代码主要功能是根据指定的开始字符来生成诗歌
        #########################################
        cur_state = sess.run(end_points['last_state'], feed_dict={
            input_data: np.array([[word_int_map[start_token]]])
        })
        while word != end_token:
            poem += word
            index = np.array([[word_int_map[word]]])
            probs, cur_state = sess.run([
                end_points['prediction'],
                end_points['last_state']
            ], feed_dict={
                input_data: index,
                end_points['initial_state']: cur_state
            })
            word = to_word(probs, vocabularies)
        #########################################
        return poem
    
#print('[INFO] train tang poem...')
#run_training_tangshi() # 训练模型
#tf.reset_default_graph()
#print('[INFO] write tang poem...')
#poem2 = gen_tangshi('月')# 生成诗歌
#print("#" * 25)
#pretty_print_poem(poem2)
#print('#' * 25)