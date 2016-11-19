"""
Goal:
    - Create RNN layers

Important Concepts/Design Choices:
    - For "rnn" it is difficult in TF to iterate over variable number of iterations based
        on the value of a tensor (I couldn't figure this out, nor could I find any
        examples of others doing this). Thus how is it possible to iterate for variable
        number of time steps for each batch? This is where the use of TF's control
        flow options come in, namely tf.while_loop. See inline comments for details.

TODO/FIX: Get iteration numbers for RNN? Scope

Credits: Adapted from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/rnn.py
"""

import tensorflow as tf

def rnn(cell, inputs, seq_lens, batch_size, embedding_dim):
    state = cell.zero_state(batch_size)

    seq_len_mask = tf.cast(tf.sequence_mask(seq_lens), tf.float32)
    # Find the maximum document length, set as total number of time steps
    time = tf.reduce_max(seq_lens)

    # Continue with loop if condition returns true
    def condition(i, inputs, state, outputs):
        return tf.less(i, time)

    # Body of while loop: runs one time step of RNN
    def body(i, inputs, state, outputs):
        # FIGURE OUT HOW TO GET PROPER NUMBERS HERE: with tf.variable_scope("Cell-Time{}".format(1)):
        with tf.variable_scope("Cell-Time"):
            # Take one time step's worth of input and create mask (time_mask discussed in rnn_cell.py)
            input_ = tf.slice(inputs, [0, i, 0], [batch_size, 1, embedding_dim])
            time_mask = tf.slice(seq_len_mask, [0, i], [batch_size, 1])
            # Squeeze to get correct dimensions - dim 1 goes to 0
            input_ = tf.squeeze(input_)

            # RNN time step
            output, state = cell(input_, state, time_mask)

            # Concatenate output to tensor of all outputs (hidden states)
            # Dimensions: batch x time x hidden_state_size
            # Concatenate along time (reshape after while_loop finishes)
            outputs = tf.concat(1, [outputs, output])
        return [tf.add(i, 1), inputs, state, outputs]

    # iterator/counter
    i = tf.constant(0)

    # initialize "outputs" arg (hidden states) to pass into while loop to vector of zeros
    # Will remove these zeros after the while loop ends
    # Did this because need to concatenate current time step's hidden state with all prev
        # timestep's hidden states; can't concatenate with "None" as one argument
    outputs = tf.TensorShape([batch_size, None])

    # Run RNN while loop
    _, _, last_state, hidden_states = tf.while_loop(condition, body, \
        loop_vars=[i, inputs, state, tf.zeros([batch_size, 1])], \
        # Shape_invariants arg allows one to specify dimensions of inputs and outputs of each iterations
        # By using "None" as a dimension, signifies that this dimension can change
        shape_invariants=[i.get_shape(), inputs.get_shape(), state.get_shape(), outputs])

    # get rid of zero output start state for concat purposes (see "body")
    hidden_states = tf.slice(hidden_states, [0, 1], [batch_size, -1])
    # reshape hidden_states to (batch x time x hidden_state_size)
    hidden_states = tf.reshape(hidden_states, [batch_size, -1, cell._state_size])

    # Dimensions: outputs (batch x time x hidden_size); last_state (batch x hidden_size)
    return (hidden_states, last_state)


def bidirectional_rnn(forward_cell, backward_cell, inputs, seq_lens, batch_size, embedding_dim, concatenate=True):
    # Reverse inputs (batch x time x embedding_dim); takes care of variable seq_len
    reverse_inputs = tf.reverse_sequence(inputs, seq_lens, seq_dim=1, batch_dim=0)

    # Run forwards and backwards RNN
    forward_outputs, forward_last_state = \
        rnn(forward_cell, inputs, seq_lens, batch_size, embedding_dim)
    backward_outputs, backward_last_state = \
        rnn(backward_cell, reverse_inputs, seq_lens, batch_size, embedding_dim)

    if concatenate:
        # last_state dimensions: batch x hidden_size
        last_state = tf.concat(1, [forward_last_state, backward_last_state])
        # outputs dimensions: batch x time x hidden_size
        outputs = tf.concat(2, [forward_outputs, backward_outputs])

        # Dimensions: outputs (batch x time x hidden_size*2); last_state (batch x hidden_size*2)
        return (outputs, last_state)

    # Dimensions: outputs (batch x time x hidden_size); last_state (batch x hidden_size)
    return (forward_outputs, forward_last_state, backward_outputs, backward_last_state)


def encoder_decoder_rnn(cell, inputs, seq_lens, batch_size, embedding_dim):
    pass
    # Run regular rnn on inputs
    # Get last state, acts as context for decoder
    # Concatenate context to decoder inputs (in testing like language model, have to go one timestep at a time)
    # Another RNN and get hidden states
    # Softmax on hidden states with loss: tf.nn.seq2seq.sequence_loss_by_example (see ptb lm usage)
    # ASK ABOUT TRAIN VS EVAL DIFFERENCE


# cell size input size should be embedding_dim_input + embedding_dim_output
def rnn_decoder_attention(cell, inputs, seq_lens, batch_size, embedding_dim, start_state, attention, attended, seq_lens_attended):
    # testing rnn decoder attention
    state = start_state

    seq_len_mask = tf.cast(tf.sequence_mask(seq_lens), tf.float32)
    # Find the maximum document length, set as total number of time steps
    time = tf.reduce_max(seq_lens)

    # Continue with loop if condition returns true
    def condition(i, inputs, state, outputs, max_attention):
        return tf.less(i, time)

    # Body of while loop: runs one time step of RNN
    def body(i, inputs, state, outputs, max_attention):
        # FIGURE OUT HOW TO GET PROPER NUMBERS HERE: with tf.variable_scope("Cell-Time{}".format(1)):
        with tf.variable_scope("Cell-Time"):
            # Take one time step's worth of input and create mask (time_mask discussed in rnn_cell.py)
            input_ = tf.slice(inputs, [0, i, 0], [batch_size, 1, embedding_dim])
            time_mask = tf.slice(seq_len_mask, [0, i], [batch_size, 1])

            alpha_weights, attend_result = \
                attention(attending=state, attended=attended, seq_lens=seq_lens_attended, batch_size=batch_size)

            # Save which word in source sentence was paid the most attention
            max_attention = tf.concat(1, [max_attention, tf.expand_dims(tf.argmax(alpha_weights, 1), 1)])

            # batch x 1 x embedding -> batch x embedding
            input_ = tf.squeeze(input_)
            context = tf.concat(1, [input_, attend_result])

            # RNN time step
            output, state = cell(context, state, time_mask)

            # Concatenate output to tensor of all outputs (hidden states)
            # Dimensions: batch x time x hidden_state_size
            # Concatenate along time (reshape after while_loop finishes)
            outputs = tf.concat(1, [outputs, output])
        return [tf.add(i, 1), inputs, state, outputs, max_attention]

    # iterator/counter
    i = tf.constant(0)
    output_dim = tf.TensorShape([batch_size, None])
    max_attention_dim = tf.TensorShape([batch_size, None])

    # Run RNN while loop
    _, _, last_state, hidden_states, max_attention = tf.while_loop(condition, body, \
        loop_vars=[i, inputs, state, tf.zeros([batch_size, 1]), tf.zeros([batch_size, 1], dtype=tf.int64)], \
        # Shape_invariants arg allows one to specify dimensions of inputs and outputs of each iterations
        # By using "None" as a dimension, signifies that this dimension can change
        shape_invariants=[i.get_shape(), inputs.get_shape(), state.get_shape(), output_dim, max_attention_dim])

    # get rid of zero `output` and `max_attention` start state for concat purposes (see "body")
    hidden_states = tf.slice(hidden_states, [0, 1], [batch_size, -1])
    # Dimensions: batch_size x batch_max_target_sentence_length - NEEDS TO BE MASKED
    max_attention = tf.slice(max_attention, [0, 1], [batch_size, -1])

    # reshape hidden_states to (batch x time x hidden_state_size)
    hidden_states = tf.reshape(hidden_states, [batch_size, -1, cell._state_size])

    # Dimensions: outputs (batch x time x hidden_size); last_state (batch x hidden_size)
    return (hidden_states, last_state, max_attention)


def encoder_decoder_attention_rnn(forward_cell, backward_cell, inputs_source, seq_lens_source, \
    decoder_cell, inputs_target, seq_lens_target, batch_size, embedding_dim, attention):
    # Run bidirectional rnn on inputs
    hidden_states_source, last_state_source = \
        bidirectional_rnn(forward_cell, backward_cell, inputs_source, seq_lens_source, batch_size, embedding_dim, concatenate=True)

    # Get last state, acts as decoder input
    # For each time step for length of input
        # Get attention weight with concatenation of correct word and hidden state over hidden states
        # Transform hidden state with weight matrix, tanh, weight matrix, softmax, get loss

    # Start state of the decoder is the output of the encoder
    # COULD ADD A TRANSFORMATION WEIGHT MATRIX HERE
    start_state = last_state_source

    # MAKE OPTION FOR DIFFERENT DIMENSION EMBEDDINGS FOR TARGET AND SOURCE
    hidden_states_target, last_state_target, max_attention = \
        rnn_decoder_attention(
            cell=decoder_cell,
            inputs=inputs_target,
            seq_lens=seq_lens_target,
            batch_size=batch_size,
            embedding_dim=embedding_dim,
            start_state=start_state,
            attention=attention,
            attended=hidden_states_source,
            seq_lens_attended=seq_lens_source
        )

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

    self.loss = tf.reduce_sum(loss_vector) / batch_size

    # transform hidden_states_target to make predictions (softmax over vocab hierarchical)
    print(hidden_states_target.get_shape())


    # Dimensions: outputs (batch x time x hidden_size); last_state (batch x hidden_size)
    return (hidden_states, last_state)

    # Save weights, argmax for alignments
