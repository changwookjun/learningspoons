# -*- coding: utf-8 -*-
import tensorflow as tf
import sys

from configs import DEFINES
import numpy as np


def layer_norm(inputs, eps=1e-6):
    # ?, 25, 128
    feature_shape = inputs.get_shape()[-1:] # 128
    mean = tf.keras.backend.mean(inputs, [-1], keepdims=True) # ?, 25, 1
    std = tf.keras.backend.std(inputs, [-1], keepdims=True) # ?, 25, 1
    beta = tf.get_variable("beta", initializer=tf.zeros(feature_shape)) # 128
    gamma = tf.get_variable("gamma", initializer=tf.ones(feature_shape)) # 128

    # ?, 25, 128
    return gamma * (inputs - mean) / (std + eps) + beta


def sublayer_connection(inputs, sublayer, dropout=0.2):
    # ?, 25, 128 / ?, 25, 128
    outputs = layer_norm(inputs + tf.keras.layers.Dropout(dropout)(sublayer))
    return outputs


def positional_encoding(dim, sentence_length):
    # 128, 25
    encoded_vec = np.array([pos/np.power(10000, 2*i/dim) 
                            for pos in range(sentence_length) for i in range(dim)])
    encoded_vec[::2] = np.sin(encoded_vec[::2]) 
    encoded_vec[1::2] = np.cos(encoded_vec[1::2])

    # 25, 128
    return tf.constant(encoded_vec.reshape([sentence_length, dim]), dtype=tf.float32)


class MultiHeadAttention(tf.keras.Model):
    def __init__(self, num_units, heads, masked=False):
        super(MultiHeadAttention, self).__init__()
        #128, 4
        self.heads = heads
        self.masked = masked

        self.query_dense = tf.keras.layers.Dense(num_units, activation=tf.nn.relu)
        self.key_dense = tf.keras.layers.Dense(num_units, activation=tf.nn.relu)
        self.value_dense = tf.keras.layers.Dense(num_units, activation=tf.nn.relu)

    def scaled_dot_product_attention(self, query, key, value, masked=False):
        # ?, 25, 32 / ?, 25, 32 / ?, 25, 32
        key_seq_length = float(key.get_shape().as_list()[-1]) # 25.0

        key = tf.transpose(key, perm=[0, 2, 1]) # ?, 32, 25
        outputs = tf.matmul(query, key) / tf.sqrt(key_seq_length) # ?, 25, 25

        if masked:
            diag_vals = tf.ones_like(outputs[0, :, :]) # 25, 25
            tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense() # 25, 25
            masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1]) # ?, 25, 25

            paddings = tf.ones_like(masks) * (-2 ** 32 + 1) # ?, 25, 25
            outputs = tf.where(tf.equal(masks, 0), paddings, outputs) # ?, 25, 25

        attention_map = tf.nn.softmax(outputs) # ?, 25, 25
        # ?, 25, 32
        return tf.matmul(attention_map, value)

    def call(self, query, key, value):
        # ?, 25, 128 / ?, 25, 128 / ?, 25, 128
        query = self.query_dense(query) # ?, 25, 128
        key = self.key_dense(key) # ?, 25, 128
        value = self.value_dense(value) # ?, 25, 128

        query = tf.concat(tf.split(query, self.heads, axis=-1), axis=0) # ?, 25, 32
        key = tf.concat(tf.split(key, self.heads, axis=-1), axis=0) # ?, 25, 32
        value = tf.concat(tf.split(value, self.heads, axis=-1), axis=0) # ?, 25, 32

        attention_map = self.scaled_dot_product_attention(query, key, value, self.masked) # ?, 25, 32
        attn_outputs = tf.concat(tf.split(attention_map, self.heads, axis=0), axis=-1) # ?, 25, 128
        return attn_outputs


class Encoder(tf.keras.Model):
    def __init__(self, model_dims, ffn_dims, attn_heads, num_layers=1):
        super(Encoder, self).__init__()
        # 128, 512, 4, 3
        self.self_attention = [MultiHeadAttention(model_dims, attn_heads) for _ in range(num_layers)]
        self.position_feedforward = [PositionWiseFeedForward(ffn_dims, model_dims) for _ in range(num_layers)]

    def call(self, inputs):
        output_layer = None
        # ?, 25, 128
        for i, (s_a, p_f) in enumerate(zip(self.self_attention, self.position_feedforward)):
            with tf.variable_scope('encoder_layer_' + str(i + 1)):
                attention_layer = sublayer_connection(inputs, s_a(inputs, inputs, inputs)) # ?, 25, 128
                output_layer = sublayer_connection(attention_layer, p_f(attention_layer)) # ?, 25, 128
                inputs = output_layer
        # ?, 25, 128
        return output_layer


class Decoder(tf.keras.Model):
    def __init__(self, model_dims, ffn_dims, attn_heads, num_layers=1):
        super(Decoder, self).__init__()
        # 128, 512, 4, 3

        self.self_attention = [MultiHeadAttention(model_dims, attn_heads, masked=True) for _ in range(num_layers)]
        self.encoder_decoder_attention = [MultiHeadAttention(model_dims, attn_heads) for _ in range(num_layers)]
        self.position_feedforward = [PositionWiseFeedForward(ffn_dims, model_dims) for _ in range(num_layers)]

    def call(self, inputs, encoder_outputs):
        output_layer = None
        # ?, 25, 128 / ?, 25, 128

        for i, (s_a, ed_a, p_f) in enumerate(zip(self.self_attention, self.encoder_decoder_attention, self.position_feedforward)):
            with tf.variable_scope('decoder_layer_' + str(i + 1)):
                masked_attention_layer = sublayer_connection(inputs, s_a(inputs, inputs, inputs)) # ?, 25, 128
                attention_layer = sublayer_connection(masked_attention_layer, ed_a(masked_attention_layer, # ?, 25, 128
                                                                                        encoder_outputs,
                                                                                        encoder_outputs))
                output_layer = sublayer_connection(attention_layer, p_f(attention_layer)) # ?, 25, 128  
                inputs = output_layer
        # ?, 25, 128
        return output_layer


class PositionWiseFeedForward(tf.keras.Model):
    def __init__(self, num_units, feature_shape):
        super(PositionWiseFeedForward, self).__init__()
        # 512, 128
        self.inner_dense = tf.keras.layers.Dense(num_units, activation=tf.nn.relu)
        self.output_dense = tf.keras.layers.Dense(feature_shape)

    def call(self, inputs):
        # ?, 25, 128
        inner_layer = self.inner_dense(inputs) # ?, 25, 512
        outputs = self.output_dense(inner_layer)  # ?, 25, 128
        return outputs


def Model(features, labels, mode, params):
    TRAIN = mode == tf.estimator.ModeKeys.TRAIN
    EVAL = mode == tf.estimator.ModeKeys.EVAL
    PREDICT = mode == tf.estimator.ModeKeys.PREDICT

    position_encode = positional_encoding(params['embedding_size'], params['max_sequence_length']) # 25, 128

    embedding = tf.keras.layers.Embedding(params['vocabulary_length'], # 12657 , 128
                                          params['embedding_size'])

    encoder_layers = Encoder(params['model_hidden_size'], params['ffn_hidden_size'],
                      params['attention_head_size'], params['layer_size'])

    decoder_layers = Decoder(params['model_hidden_size'], params['ffn_hidden_size'],
                      params['attention_head_size'], params['layer_size'])

    logit_layer = tf.keras.layers.Dense(params['vocabulary_length'])

    with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE):
        x_embedded_matrix = embedding(features['input']) + position_encode #?, 25, 128
        
        encoder_outputs = encoder_layers(x_embedded_matrix) # ?, 25, 128

    loop_count = params['max_sequence_length'] if PREDICT else 1 # 1

    predict, output, logits = None, None, None

    predict_tokens = list()
    decoder_input = [params['max_sequence_length']]
    output = tf.expand_dims(decoder_input, 0)
    print("loop_count: ", loop_count)

    for i in range(loop_count):
        with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
            print("i: ", i)
            if i > 0:
                output = tf.concat([tf.ones((output.shape[0], 1), dtype=tf.int64), predict[:, :-1]], axis=-1)
            else:
                output = features['output'] # ?, 25

            print("output: ", output)
            y_embedded_matrix = embedding(output) + position_encode # ?, 25, 128
            decoder_outputs = decoder_layers(y_embedded_matrix, encoder_outputs) # ?, 25, 128
            
            logits = logit_layer(decoder_outputs) # ?, 25, 12657
            print("logits: ", logits)
            predict = tf.argmax(logits, 2) # ?, 25
            print("predict: ", predict)

    if PREDICT:
        predict_tokens = list()
        predict_tokens.append(predict)
        predict = tf.stack(predict_tokens, axis=0)
        predictions = {
            'indexs': predict,
            'logits': logits,
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    accuracy = tf.metrics.accuracy(labels=labels, predictions=predict, name='accOp')

    metrics = {'accuracy': accuracy}
    tf.summary.scalar('accuracy', accuracy[1])

    if EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)

    assert TRAIN

    optimizer = tf.train.AdamOptimizer(learning_rate=params['learning_rate'])
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)