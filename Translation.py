import tensorflow as tf
import numpy as np

from rnn_cell import GRUCell
from rnn import bidirectional_rnn, rnn_decoder_attention

class Translation(object):
	def __init__(self, hidden_size=128, vocab_size=50000, embedding_dim=100, batch_size=32):

		tf.set_random_seed(1234)

        # Placeholders
        # can add assert statements to ensure shared None dimensions are equal (batch_size)
        self.seq_lens_target = tf.placeholder(tf.int32, [None, ], name="seq_lens_target")
        self.seq_lens_source = tf.placeholder(tf.int32, [None, ], name="seq_lens_source")
        self.input_target = tf.placeholder(tf.int32, [None, None], name="input_target")
        self.input_source = tf.placeholder(tf.int32, [None, None], name="input_source")

        mask_target = tf.cast(tf.sequence_mask(self.seq_lens_target), tf.int32)
        mask_source = tf.cast(tf.sequence_mask(self.seq_lens_source), tf.int32)

        masked_target = tf.mul(self.input_target, mask_target)
        masked_source = tf.mul(self.input_source, mask_source)

        # Buildling Graph (Network Layers)
		# ==================================================
        with tf.variable_scope("embedding"):
            W_embeddings_target = tf.get_variable(shape=[vocab_size, embedding_dim], \
                                           initializer=tf.random_uniform_initializer(-0.01, 0.01),\
                                           name="W_embeddings_target")
            W_embeddings_source = tf.get_variable(shape=[vocab_size, embedding_dim], \
                                           initializer=tf.random_uniform_initializer(-0.01, 0.01),\
                                           name="W_embeddings_source")
            ################## Make option to use pre-trained embeddings ##################

            # Dimensions: batch x max_length x embedding_dim
            target_embedding = tf.gather(W_embeddings_target, masked_target)
            source_embedding = tf.gather(W_embeddings_source, masked_source)

        with tf.variable_scope("encoder"):
            forward_cell = GRUCell(state_size=hidden_size, input_size=embedding_dim, scope="forward-GRU")
            backward_cell = GRUCell(state_size=hidden_size, input_size=embedding_dim, scope="backward-GRU")

            hidden_states_source, last_state_source = \
                bidirectional_rnn(forward_cell, backward_cell, inputs_source, \
                    self.seq_lens_source, batch_size, embedding_dim, concatenate=True)

        with tf.variable_scope("decoder"):
            decoder_cell = GRUCell(state_size=hidden_size, input_size=embedding_dim, scope="decoder-GRU")

            # Start state of the decoder is the output of the encoder
            # COULD ADD A TRANSFORMATION WEIGHT MATRIX HERE
            start_state = last_state_source

            # MAKE OPTION FOR DIFFERENT DIMENSION EMBEDDINGS FOR TARGET AND SOURCE
            hidden_states_target, last_state_target, max_attention = \
                rnn_decoder_attention(
                    cell=decoder_cell,
                    inputs=inputs_target,
                    seq_lens=self.seq_lens_target,
                    batch_size=batch_size,
                    embedding_dim=embedding_dim,
                    start_state=start_state,
                    attention=attention,
                    attended=hidden_states_source,
                    seq_lens_attended=self.seq_lens_source
                )

        with tf.variable_scope("loss"):
            # reshape / mask labels and outputs
            labels = tf.reshape(tf.boolean_mask(masked_target, tf.sequence_mask(self.seq_lens_target)), [-1, 1])
            hidden_states = tf.boolean_mask(hidden_states_target, tf.sequence_mask(self.seq_lens_target))

            # MAX ATTENTION NEEDS TO BE MASKED
            W_softmax = tf.get_variable(shape=[vocab_size, hidden_size*2], \
                                           initializer=tf.random_uniform_initializer(-0.01, 0.01),\
                                           name="W_softmax")
            b_softmax = tf.get_variable(name="softmax_bias", shape=[vocab_size], \
                initializer=tf.constant_initializer(0.0))

            # calculate loss
            loss_vector = tf.nn.sampled_softmax_loss(
                weights=W_softmax,
                biases=b_softmax,
                inputs=hidden_states,
                labels=labels,
                num_sampled=100,
                num_classes=vocab_size,
                num_true=1,
                remove_accidental_hits=True,
                name='sampled_softmax_loss'
            )

            self.loss = tf.reduce_sum(loss_vector) / batch_size
