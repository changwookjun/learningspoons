# -*- coding: utf-8 -*-
import tensorflow as tf
import sys

from configs import DEFINES
import numpy as np


def layer_norm(inputs, eps=1e-6):
    print("layer_norm inputs: ", inputs)
    # LayerNorm(x + Sublayer(x))
    feature_shape = inputs.get_shape()[-1:]
    print("layer_norm feature_shape: ", feature_shape)
    #  평균과 표준편차을 넘겨 준다.
    mean = tf.keras.backend.mean(inputs, [-1], keepdims=True)
    print("layer_norm mean: ", mean)
    std = tf.keras.backend.std(inputs, [-1], keepdims=True)
    print("layer_norm std: ", std)
    beta = tf.get_variable("beta", initializer=tf.zeros(feature_shape))
    print("layer_norm beta: ", beta)
    gamma = tf.get_variable("gamma", initializer=tf.ones(feature_shape))
    print("layer_norm gamma: ", gamma)

    print("layer_norm return: ", gamma * (inputs - mean) / (std + eps) + beta)
    return gamma * (inputs - mean) / (std + eps) + beta


def sublayer_connection(inputs, sublayer, dropout=0.2):
    print("sublayer_connection inputs: ", inputs)
    print("sublayer_connection sublayer: ", sublayer)
    outputs = layer_norm(inputs + tf.keras.layers.Dropout(dropout)(sublayer))
    print("sublayer_connection outputs: ", outputs)
    return outputs


def positional_encoding(dim, sentence_length):
    print("positional_encoding dim: ", dim)
    print("positional_encoding sentence_length: ", sentence_length)
    encoded_vec = np.array([pos/np.power(10000, 2*i/dim)
                            for pos in range(sentence_length) for i in range(dim)])
    print("positional_encoding encoded_vec: ", encoded_vec)
    encoded_vec[::2] = np.sin(encoded_vec[::2])
    print("positional_encoding encoded_vec[::2]: ", encoded_vec[::2])
    encoded_vec[1::2] = np.cos(encoded_vec[1::2])
    print("positional_encoding encoded_vec[1::2]: ", encoded_vec[1::2])

    print("positional_encoding return: ", tf.constant(encoded_vec.reshape([sentence_length, dim]), dtype=tf.float32))
    return tf.constant(encoded_vec.reshape([sentence_length, dim]), dtype=tf.float32)


class MultiHeadAttention(tf.keras.Model):
    def __init__(self, num_units, heads, masked=False):
        print("MultiHeadAttention __init__ num_units: ", num_units)
        print("MultiHeadAttention __init__ heads: ", heads)
        super(MultiHeadAttention, self).__init__()

        self.heads = heads
        self.masked = masked

        self.query_dense = tf.keras.layers.Dense(num_units, activation=tf.nn.relu)
        print("MultiHeadAttention __init__ query_dense: ", query_dense)
        self.key_dense = tf.keras.layers.Dense(num_units, activation=tf.nn.relu)
        print("MultiHeadAttention __init__ key_dense: ", key_dense)
        self.value_dense = tf.keras.layers.Dense(num_units, activation=tf.nn.relu)
        print("MultiHeadAttention __init__ value_dense: ", value_dense)

    def scaled_dot_product_attention(self, query, key, value, masked=False):
        print("MultiHeadAttention scaled_dot_product_attention query: ", query)
        print("MultiHeadAttention scaled_dot_product_attention key: ", key)
        print("MultiHeadAttention scaled_dot_product_attention value: ", value)
        key_seq_length = float(key.get_shape().as_list()[-2])
        print("MultiHeadAttention scaled_dot_product_attention key_seq_length: ", key_seq_length)
        key = tf.transpose(key, perm=[0, 2, 1])
        print("MultiHeadAttention scaled_dot_product_attention key: ", key)
        outputs = tf.matmul(query, key) / tf.sqrt(key_seq_length)
        print("MultiHeadAttention scaled_dot_product_attention outputs: ", outputs)

        if masked:
            diag_vals = tf.ones_like(outputs[0, :, :])
            print("MultiHeadAttention scaled_dot_product_attention diag_vals: ", diag_vals)
            tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()
            print("MultiHeadAttention scaled_dot_product_attention tril: ", tril)
            masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1])
            print("MultiHeadAttention scaled_dot_product_attention masks: ", masks)

            paddings = tf.ones_like(masks) * (-2 ** 32 + 1)
            print("MultiHeadAttention scaled_dot_product_attention paddings: ", paddings)
            outputs = tf.where(tf.equal(masks, 0), paddings, outputs)
            print("MultiHeadAttention scaled_dot_product_attention outputs: ", outputs)

        attention_map = tf.nn.softmax(outputs)
        print("MultiHeadAttention scaled_dot_product_attention attention_map: ", attention_map)

        print("MultiHeadAttention scaled_dot_product_attention return: ", tf.matmul(attention_map, value))
        return tf.matmul(attention_map, value)

    def call(self, query, key, value):
        print("MultiHeadAttention call query: ", query)
        print("MultiHeadAttention call key: ", key)
        print("MultiHeadAttention call value: ", value)
        query = self.query_dense(query)
        print("MultiHeadAttention call query: ", query)
        key = self.key_dense(key)
        print("MultiHeadAttention call key: ", key)
        value = self.value_dense(value)
        print("MultiHeadAttention call value: ", value)

        query = tf.concat(tf.split(query, self.heads, axis=-1), axis=0)
        print("MultiHeadAttention call query: ", query)
        key = tf.concat(tf.split(key, self.heads, axis=-1), axis=0)
        print("MultiHeadAttention call key: ", key)
        value = tf.concat(tf.split(value, self.heads, axis=-1), axis=0)
        print("MultiHeadAttention call value: ", value)

        attention_map = self.scaled_dot_product_attention(query, key, value, self.masked)
        print("MultiHeadAttention call attention_map: ", attention_map)
        attn_outputs = tf.concat(tf.split(attention_map, self.heads, axis=0), axis=-1)
        print("MultiHeadAttention call attn_outputs: ", attn_outputs)
        return attn_outputs


class Encoder(tf.keras.Model):
    def __init__(self, model_dims, ffn_dims, attn_heads, num_layers=1):
        super(Encoder, self).__init__()
        print("Encoder __init__ model_dims: ", model_dims)
        print("Encoder __init__ ffn_dims: ", ffn_dims)
        print("Encoder __init__ attn_heads: ", attn_heads)
        print("Encoder __init__ num_layers: ", num_layers)
        self.self_attention = [MultiHeadAttention(model_dims, attn_heads) for _ in range(num_layers)]
        print("Encoder __init__ self_attention: ", self_attention)
        self.position_feedforward = [PositionWiseFeedForward(ffn_dims, model_dims) for _ in range(num_layers)]
        print("Encoder __init__ position_feedforward: ", position_feedforward)

    def call(self, inputs):
        output_layer = None
        print("Encoder call inputs: ", inputs)
        for i, (s_a, p_f) in enumerate(zip(self.self_attention, self.position_feedforward)):
            print("Encoder call i: ", i)
            print("Encoder call s_a: ", s_a)
            print("Encoder call p_f: ", p_f)
            with tf.variable_scope('encoder_layer_' + str(i + 1)):
                attention_layer = sublayer_connection(inputs, s_a(inputs, inputs, inputs))
                print("Encoder call attention_layer: ", attention_layer)
                output_layer = sublayer_connection(attention_layer, p_f(attention_layer))
                print("Encoder call output_layer: ", output_layer)
                inputs = output_layer
                print("Encoder call inputs: ", inputs)
        print("Encoder call return: ", output_layer)
        return output_layer


class Decoder(tf.keras.Model):
    def __init__(self, model_dims, ffn_dims, attn_heads, num_layers=1):
        super(Decoder, self).__init__()
        print("Decoder __init__ model_dims: ", model_dims)
        print("Decoder __init__ ffn_dims: ", ffn_dims)
        print("Decoder __init__ attn_heads: ", attn_heads)
        print("Decoder __init__ num_layers: ", num_layers)

        self.self_attention = [MultiHeadAttention(model_dims, attn_heads, masked=True) for _ in range(num_layers)]
        print("Decoder __init__ self_attention: ", self_attention)
        self.encoder_decoder_attention = [MultiHeadAttention(model_dims, attn_heads) for _ in range(num_layers)]
        print("Decoder __init__ encoder_decoder_attention: ", encoder_decoder_attention)
        self.position_feedforward = [PositionWiseFeedForward(ffn_dims, model_dims) for _ in range(num_layers)]
        print("Decoder __init__ position_feedforward: ", position_feedforward)

    def call(self, inputs, encoder_outputs):
        output_layer = None
        print("Decoder call inputs: ", inputs)
        print("Decoder call encoder_outputs: ", encoder_outputs)

        for i, (s_a, ed_a, p_f) in enumerate(zip(self.self_attention, self.encoder_decoder_attention, self.position_feedforward)):
            print("Decoder call i: ", i)
            print("Decoder call s_a: ", s_a)
            print("Decoder call ed_a: ", ed_a)
            print("Decoder call p_f: ", p_f)
            with tf.variable_scope('decoder_layer_' + str(i + 1)):
                masked_attention_layer = sublayer_connection(inputs, s_a(inputs, inputs, inputs))
                print("Decoder call masked_attention_layer: ", masked_attention_layer)
                attention_layer = sublayer_connection(masked_attention_layer, ed_a(masked_attention_layer,
                print("Decoder call attention_layer: ", attention_layer)                                                                           encoder_outputs,
                                                                                           encoder_outputs))
                output_layer = sublayer_connection(attention_layer, p_f(attention_layer))
                print("Decoder call output_layer: ", output_layer)   
                inputs = output_layer
        print("Decoder call return: ", output_layer) 
        return output_layer


class PositionWiseFeedForward(tf.keras.Model):
    def __init__(self, num_units, feature_shape):
        super(PositionWiseFeedForward, self).__init__()
        print("PositionWiseFeedForward __init__ num_units: ", num_units) 
        print("PositionWiseFeedForward __init__ feature_shape: ", feature_shape) 
        self.inner_dense = tf.keras.layers.Dense(num_units, activation=tf.nn.relu)
        print("PositionWiseFeedForward __init__ inner_dense: ", inner_dense)
        self.output_dense = tf.keras.layers.Dense(feature_shape)
        print("PositionWiseFeedForward __init__ output_dense: ", output_dense)

    def call(self, inputs):
        print("PositionWiseFeedForward call inputs: ", inputs)
        inner_layer = self.inner_dense(inputs)
        print("PositionWiseFeedForward call inner_layer: ", inner_layer)
        outputs = self.output_dense(inner_layer)
        print("PositionWiseFeedForward call outputs: ", outputs)
        return outputs


def Model(features, labels, mode, params):
    TRAIN = mode == tf.estimator.ModeKeys.TRAIN
    EVAL = mode == tf.estimator.ModeKeys.EVAL
    PREDICT = mode == tf.estimator.ModeKeys.PREDICT

    position_encode = positional_encoding(params['embedding_size'], params['max_sequence_length'])
    print("Model position_encode: ", position_encode)
    embedding = tf.keras.layers.Embedding(params['vocabulary_length'],
                                          params['embedding_size'])
    print("Model embedding: ", embedding)

    encoder_layers = Encoder(params['model_hidden_size'], params['ffn_hidden_size'],
                      params['attention_head_size'], params['layer_size'])
    print("Model encoder_layers: ", encoder_layers)

    decoder_layers = Decoder(params['model_hidden_size'], params['ffn_hidden_size'],
                      params['attention_head_size'], params['layer_size'])
    print("Model decoder_layers: ", decoder_layers)

    logit_layer = tf.keras.layers.Dense(params['vocabulary_length'])
    print("Model logit_layer: ", logit_layer)

    with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE):
        x_embedded_matrix = embedding(features['input']) + position_encode
        print("Model x_embedded_matrix: ", x_embedded_matrix)
        encoder_outputs = encoder_layers(x_embedded_matrix)
        print("Model encoder_outputs: ", encoder_outputs)

    loop_count = params['max_sequence_length'] if PREDICT else 1
     print("Model loop_count: ", loop_count)
    predict, output, logits = None, None, None

    for i in range(loop_count):
        print("Model i: ", i)
        with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
            if i > 0:
                output = tf.concat([tf.ones((output.shape[0], 1), dtype=tf.int64), predict[:, :-1]], axis=-1)
                print("Model output: ", output)
            else:
                output = features['output']
                print("Model output: ", output)

            y_embedded_matrix = embedding(output) + position_encode
            print("Model y_embedded_matrix: ", y_embedded_matrix)
            decoder_outputs = decoder_layers(y_embedded_matrix, encoder_outputs)
            print("Model decoder_outputs: ", decoder_outputs)

            logits = logit_layer(decoder_outputs)
            print("Model logits: ", logits)
            predict = tf.argmax(logits, 2)
            print("Model predict: ", predict)

    if PREDICT:
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