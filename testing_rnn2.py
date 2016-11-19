import tensorflow as tf

from rnn_cell import GRUCell
from rnn import bidirectional_rnn, rnn_decoder_attention
#from rnn import encoder_decoder_attention_rnn
from attention import BilinearFunction
import pickle

# parameters
hidden_size = 13
embedding_dim = 10
batch_size = 2
vocab_size = 100

# Starting interactive Session
sess = tf.InteractiveSession()

seq_lens_target = tf.placeholder(tf.int32, [None, ], name="seq_lens_target")
seq_lens_source = tf.placeholder(tf.int32, [None, ], name="seq_lens_source")
input_target = tf.placeholder(tf.int32, [None, None], name="input_target")
input_source = tf.placeholder(tf.int32, [None, None], name="input_source")

feed = {
    seq_lens_target: [5,4],
    seq_lens_source: [2,3],
    input_target: [[20,30,40,50,60],[2,3,4,5,-1]],
    input_source: [[2,3,-1],[1,2,3]],
}

mask_target = tf.cast(tf.sequence_mask(seq_lens_target), tf.int32)
mask_source = tf.cast(tf.sequence_mask(seq_lens_source), tf.int32)

masked_target = tf.mul(input_target, mask_target)
masked_source = tf.mul(input_source, mask_source)

with tf.variable_scope("embedding"):
    W_embeddings_target = tf.get_variable(shape=[vocab_size, embedding_dim], \
                                   initializer=tf.random_uniform_initializer(-0.01, 0.01),\
                                   name="W_embeddings_target")
    W_embeddings_source = tf.get_variable(shape=[vocab_size, embedding_dim], \
                                   initializer=tf.random_uniform_initializer(-0.01, 0.01),\
                                   name="W_embeddings_source")

    # Dimensions: batch x max_length x embedding_dim
    target_embedding = tf.gather(W_embeddings_target, masked_target)
    source_embedding = tf.gather(W_embeddings_source, masked_source)

with tf.variable_scope("rnn"):
    # Layers
    forward_cell = GRUCell(state_size=hidden_size, input_size=embedding_dim, scope="forward-GRU")
    backward_cell = GRUCell(state_size=hidden_size, input_size=embedding_dim, scope="backward-GRU")
    decoder_cell = GRUCell(state_size=hidden_size, input_size=embedding_dim)

    hidden_states_source, last_state_source = \
        bidirectional_rnn(forward_cell, backward_cell, source_embedding, seq_lens_source, batch_size, embedding_dim, concatenate=True)

    start_state = last_state_source
    seq_lens = seq_lens_target
    attention = BilinearFunction(attending_size=hidden_size*2, attended_size=hidden_size*2) # DIMENSIONS!!!!
    inputs = target_embedding
    attended = hidden_states_source
    seq_lens_attended = seq_lens_source
    cell = GRUCell(state_size=hidden_size*2, input_size=embedding_dim+hidden_size*2, scope="decoder-cell")

    hidden_states_target, last_state_target, max_attention = \
        rnn_decoder_attention(cell, inputs, seq_lens, batch_size, embedding_dim, start_state, attention, attended, seq_lens_attended)

    # reshape / mask labels and outputs
    labels = tf.reshape(tf.boolean_mask(masked_target, tf.sequence_mask(seq_lens_target)), [-1, 1])
    hidden_states = tf.boolean_mask(hidden_states_target, tf.sequence_mask(seq_lens_target))

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

    loss = tf.reduce_sum(loss_vector) / batch_size

"""
encoder_decoder_attention_rnn(
    forward_cell=forward_cell,
    backward_cell=backward_cell,
    inputs_source=source_embedding,
    seq_lens_source=seq_lens_source,
    decoder_cell=deocder_cell,
    inputs_target=target_embedding,
    seq_lens_target=seq_lens_target,
    batch_size=batch_size,
    embedding_dim=embedding_dim,
    attention=attention
)
"""

sess.run(tf.initialize_all_variables())

#print(hidden_states_source.eval(feed))
#print(max_attention.eval(feed))
print(labels.eval(feed))
#print(hidden_states.eval(feed))
#print(hidden_states.get_shape())
#print(hidden_states_target.get_shape())
print(loss.eval(feed))

sess.close()
