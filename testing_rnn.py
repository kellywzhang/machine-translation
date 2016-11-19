import tensorflow as tf

from rnn_cell import GRUCell
from rnn import bidirectional_rnn
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

    # DIMENSIONS!!!!
    #attention = BilinearFunction(attending_size=hidden_size*2, attended_size=hidden_size*2)

    hidden_states_source, last_state_source = \
        bidirectional_rnn(forward_cell, backward_cell, source_embedding, seq_lens_source, batch_size, embedding_dim, concatenate=True)

    start_state = last_state_source
    seq_lens = seq_lens_target
    attention = BilinearFunction(attending_size=hidden_size*2, attended_size=hidden_size*2)
    inputs = target_embedding
    attended = hidden_states_source
    seq_lens_attended = seq_lens_source
    cell = GRUCell(state_size=hidden_size*2, input_size=embedding_dim+hidden_size*2, scope="decoder-cell")

    # testing rnn decoder attention
    state = start_state

    seq_len_mask = tf.cast(tf.sequence_mask(seq_lens), tf.float32)
    # Find the maximum document length, set as total number of time steps
    time = tf.reduce_max(seq_lens)

    # Continue with loop if condition returns true
    def condition(i, inputs, state, outputs, max_attention, alpha_weights_save):
        return tf.less(i, time)

    # Body of while loop: runs one time step of RNN
    def body(i, inputs, state, outputs, max_attention, alpha_weights_save):
        # FIGURE OUT HOW TO GET PROPER NUMBERS HERE: with tf.variable_scope("Cell-Time{}".format(1)):
        with tf.variable_scope("Cell-Time"):
            # Take one time step's worth of input and create mask (time_mask discussed in rnn_cell.py)
            input_ = tf.slice(inputs, [0, i, 0], [batch_size, 1, embedding_dim])
            time_mask = tf.slice(seq_len_mask, [0, i], [batch_size, 1])

            alpha_weights, attend_result = \
                attention(attending=state, attended=attended, seq_lens=seq_lens_attended, batch_size=batch_size)

            #print(alpha_weights)
            #print(tf.argmax(alpha_weights, 1))
            max_attention = tf.concat(1, [max_attention, tf.expand_dims(tf.argmax(alpha_weights, 1), 1)])

            # batch x 1 x embedding -> batch x embedding
            input_ = tf.squeeze(input_)
            context = tf.concat(1, [input_, attend_result])

            # RNN time step
            output, state = cell(context, state, time_mask)

            alpha_weights = tf.expand_dims(alpha_weights, 2)
            print(alpha_weights.get_shape())
            alpha_weights_save = tf.concat(2, [alpha_weights_save, alpha_weights])

            # Concatenate output to tensor of all outputs (hidden states)
            # Dimensions: batch x time x hidden_state_size
            # Concatenate along time (reshape after while_loop finishes)
            outputs = tf.concat(1, [outputs, output])
        return [tf.add(i, 1), inputs, state, outputs, max_attention, alpha_weights_save]

    # iterator/counter
    i = tf.constant(0)
    output_dim = tf.TensorShape([batch_size, None])
    max_attention_dim = tf.TensorShape([batch_size, None])
    alpha_weights_save_dim = tf.TensorShape([None, None, None])

# max attention needs target mask
# get rid of zero first state

    # Run RNN while loop
    _, _, last_state, hidden_states, max_attention, attend_r = tf.while_loop(condition, body, \
        loop_vars=[i, inputs, state, tf.zeros([batch_size, 1]), tf.zeros([batch_size, 1], dtype=tf.int64), tf.zeros([batch_size, 3, 1])], \
        # Shape_invariants arg allows one to specify dimensions of inputs and outputs of each iterations
        # By using "None" as a dimension, signifies that this dimension can change
        shape_invariants=[i.get_shape(), inputs.get_shape(), state.get_shape(), output_dim, max_attention_dim, alpha_weights_save_dim])

    # get rid of zero output start state for concat purposes (see "body")
    hidden_states = tf.slice(hidden_states, [0, 1], [batch_size, -1])
    max_attention = tf.slice(max_attention, [0, 1], [batch_size, -1])
    # reshape hidden_states to (batch x time x hidden_state_size)
    hidden_states = tf.reshape(hidden_states, [batch_size, -1, cell._state_size])

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
print(max_attention.eval(feed))
print(attend_r.eval(feed))

sess.close()
