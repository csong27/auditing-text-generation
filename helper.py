import numpy as np
import keras.backend as K
from keras.layers import Layer
from keras.legacy import interfaces
from keras.engine import InputSpec
from keras import activations, initializers, regularizers, constraints
from keras.layers.recurrent import RNN

from collections import namedtuple


def words_to_indices(data, vocab):
    return [[vocab[w] for w in t] for t in data]


def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)

    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)

        yield inputs[excerpt], targets[excerpt]


def flatten_data(data):
    return np.asarray([w for t in data for w in t]).astype(np.int32)


class _CuDNNRNN(RNN):
    def __init__(self,
                 return_sequences=False,
                 return_state=False,
                 go_backwards=False,
                 stateful=False,
                 **kwargs):
        if K.backend() != 'tensorflow':
            raise RuntimeError('CuDNN RNNs are only available '
                               'with the TensorFlow backend.')
        super(RNN, self).__init__(**kwargs)
        self.return_sequences = return_sequences
        self.return_state = return_state
        self.go_backwards = go_backwards
        self.stateful = stateful
        self.supports_masking = False
        self.input_spec = [InputSpec(ndim=3)]
        if hasattr(self.cell.state_size, '__len__'):
            state_size = self.cell.state_size
        else:
            state_size = [self.cell.state_size]
        self.state_spec = [InputSpec(shape=(None, dim))
                           for dim in state_size]
        self.constants_spec = None
        self._states = None
        self._num_constants = None

    def _canonical_to_params(self, weights, biases):
        import tensorflow as tf
        weights = [tf.reshape(x, (-1,)) for x in weights]
        biases = [tf.reshape(x, (-1,)) for x in biases]
        return tf.concat(weights + biases, 0)

    def call(self, inputs, mask=None, training=None, initial_state=None):
        if isinstance(mask, list):
            mask = mask[0]
        if mask is not None:
            raise ValueError('Masking is not supported for CuDNN RNNs.')

        # input shape: `(samples, time (padded with zeros), input_dim)`
        # note that the .build() method of subclasses MUST define
        # self.input_spec and self.state_spec with complete input shapes.
        if isinstance(inputs, list):
            initial_state = inputs[1:]
            inputs = inputs[0]
        elif initial_state is not None:
            pass
        elif self.stateful:
            initial_state = self.states
        else:
            initial_state = self.get_initial_state(inputs)

        if len(initial_state) != len(self.states):
            raise ValueError('Layer has ' + str(len(self.states)) +
                             ' states but was passed ' +
                             str(len(initial_state)) +
                             ' initial states.')

        if self.go_backwards:
            # Reverse time axis.
            inputs = K.reverse(inputs, 1)
        output, states = self._process_batch(inputs, initial_state, training)

        if self.stateful:
            updates = []
            for i in range(len(states)):
                updates.append((self.states[i], states[i]))
            self.add_update(updates, inputs)

        if self.return_state:
            return [output] + states
        else:
            return output

    def get_config(self):
        config = {'return_sequences': self.return_sequences,
                  'return_state': self.return_state,
                  'go_backwards': self.go_backwards,
                  'stateful': self.stateful}
        base_config = super(RNN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    @property
    def trainable_weights(self):
        if self.trainable and self.built:
            return [self.kernel, self.recurrent_kernel, self.bias]
        return []

    @property
    def non_trainable_weights(self):
        if not self.trainable and self.built:
            return [self.kernel, self.recurrent_kernel, self.bias]
        return []

    @property
    def losses(self):
        return super(RNN, self).losses

    def get_losses_for(self, inputs=None):
        return super(RNN, self).get_losses_for(inputs=inputs)


class CuDNNLSTM(_CuDNNRNN):
    def __init__(self, units,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 unit_forget_bias=True,
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 return_sequences=False,
                 return_state=False,
                 stateful=False,
                 dropout=0.,
                 **kwargs):
        self.units = units
        super(CuDNNLSTM, self).__init__(
            return_sequences=return_sequences,
            return_state=return_state,
            stateful=stateful,
            **kwargs)

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.unit_forget_bias = unit_forget_bias

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.dropout = dropout

    @property
    def cell(self):
        Cell = namedtuple('cell', 'state_size')
        cell = Cell(state_size=(self.units, self.units))
        return cell

    def build(self, input_shape):
        super(CuDNNLSTM, self).build(input_shape)
        if isinstance(input_shape, list):
            input_shape = input_shape[0]
        input_dim = input_shape[-1]

        from tensorflow.contrib.cudnn_rnn.python.ops import cudnn_rnn_ops
        self._cudnn_lstm = cudnn_rnn_ops.CudnnLSTM(
            num_layers=1,
            num_units=self.units,
            input_size=input_dim,
            input_mode='linear_input',
            dropout=self.dropout,
        )

        self.kernel = self.add_weight(shape=(input_dim, self.units * 4),
                                      name='kernel',
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units * 4),
            name='recurrent_kernel',
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint)

        if self.unit_forget_bias:
            def bias_initializer(shape, *args, **kwargs):
                return K.concatenate([
                    self.bias_initializer((self.units * 5,), *args, **kwargs),
                    initializers.Ones()((self.units,), *args, **kwargs),
                    self.bias_initializer((self.units * 2,), *args, **kwargs),
                ])
        else:
            bias_initializer = self.bias_initializer
        self.bias = self.add_weight(shape=(self.units * 8,),
                                    name='bias',
                                    initializer=bias_initializer,
                                    regularizer=self.bias_regularizer,
                                    constraint=self.bias_constraint)

        self.kernel_i = self.kernel[:, :self.units]
        self.kernel_f = self.kernel[:, self.units: self.units * 2]
        self.kernel_c = self.kernel[:, self.units * 2: self.units * 3]
        self.kernel_o = self.kernel[:, self.units * 3:]

        self.recurrent_kernel_i = self.recurrent_kernel[:, :self.units]
        self.recurrent_kernel_f = self.recurrent_kernel[:, self.units: self.units * 2]
        self.recurrent_kernel_c = self.recurrent_kernel[:, self.units * 2: self.units * 3]
        self.recurrent_kernel_o = self.recurrent_kernel[:, self.units * 3:]

        self.bias_i_i = self.bias[:self.units]
        self.bias_f_i = self.bias[self.units: self.units * 2]
        self.bias_c_i = self.bias[self.units * 2: self.units * 3]
        self.bias_o_i = self.bias[self.units * 3: self.units * 4]
        self.bias_i = self.bias[self.units * 4: self.units * 5]
        self.bias_f = self.bias[self.units * 5: self.units * 6]
        self.bias_c = self.bias[self.units * 6: self.units * 7]
        self.bias_o = self.bias[self.units * 7:]

        self.built = True

    def _process_batch(self, inputs, initial_state, training):
        if training is None:
            training = K.learning_phase()

        import tensorflow as tf
        inputs = tf.transpose(inputs, (1, 0, 2))
        input_h = initial_state[0]
        input_c = initial_state[1]
        input_h = tf.expand_dims(input_h, axis=0)
        input_c = tf.expand_dims(input_c, axis=0)

        params = self._canonical_to_params(
            weights=[
                self.kernel_i,
                self.kernel_f,
                self.kernel_c,
                self.kernel_o,
                self.recurrent_kernel_i,
                self.recurrent_kernel_f,
                self.recurrent_kernel_c,
                self.recurrent_kernel_o,
            ],
            biases=[
                self.bias_i_i,
                self.bias_f_i,
                self.bias_c_i,
                self.bias_o_i,
                self.bias_i,
                self.bias_f,
                self.bias_c,
                self.bias_o,
            ],
        )
        outputs, h, c = self._cudnn_lstm(
            inputs,
            input_h=input_h,
            input_c=input_c,
            params=params,
            is_training=training)

        if self.stateful or self.return_state:
            h = h[0]
            c = c[0]
        if self.return_sequences:
            output = tf.transpose(outputs, (1, 0, 2))
        else:
            output = outputs[-1]
        return output, [h, c]

    def get_config(self):
        config = {
            'units': self.units,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'recurrent_initializer': initializers.serialize(self.recurrent_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'unit_forget_bias': self.unit_forget_bias,
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'recurrent_regularizer': regularizers.serialize(self.recurrent_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'recurrent_constraint': constraints.serialize(self.recurrent_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)}
        base_config = super(CuDNNLSTM, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Attention(Layer):
    def __init__(self, units,
                 activation='linear',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(Attention, self).__init__(**kwargs)
        self.units = units
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        # self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True

    def build(self, input_shape):
        if not isinstance(input_shape, list) or len(input_shape) != 2:
            raise ValueError('An attention layer should be called '
                             'on a list of 2 inputs.')
        enc_dim = input_shape[0][-1]
        dec_dim = input_shape[1][-1]

        self.W_enc = self.add_weight(shape=(enc_dim, self.units),
                                      initializer=self.kernel_initializer,
                                      name='W_enc',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)

        self.W_dec = self.add_weight(shape=(dec_dim, self.units),
                                      initializer=self.kernel_initializer,
                                      name='W_dec',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)

        self.W_score = self.add_weight(shape=(self.units, 1),
                                      initializer=self.kernel_initializer,
                                      name='W_score',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)

        if self.use_bias:
            self.bias_enc = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        name='bias_enc',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
            self.bias_dec = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        name='bias_dec',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
            self.bias_score = self.add_weight(shape=(1,),
                                        initializer=self.bias_initializer,
                                        name='bias_score',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)

        else:
            self.bias_enc = None
            self.bias_dec = None
            self.bias_score = None

        # self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True

    def call(self, inputs, **kwargs):
        if not isinstance(inputs, list) or len(inputs) != 2:
            raise ValueError('An attention layer should be called '
                             'on a list of 2 inputs.')
        encodings, decodings = inputs
        d_enc = K.dot(encodings, self.W_enc)
        d_dec = K.dot(decodings, self.W_dec)

        if self.use_bias:
            d_enc = K.bias_add(d_enc, self.bias_enc)
            d_dec = K.bias_add(d_dec, self.bias_dec)

        if self.activation is not None:
            d_enc = self.activation(d_enc)
            d_dec = self.activation(d_dec)

        enc_seqlen = K.shape(d_enc)[1]
        d_dec_shape = K.shape(d_dec)

        stacked_d_dec = K.tile(d_dec, [enc_seqlen, 1, 1])  # enc time x batch x dec time  x da
        stacked_d_dec = K.reshape(stacked_d_dec, [enc_seqlen, d_dec_shape[0], d_dec_shape[1], d_dec_shape[2]])
        stacked_d_dec = K.permute_dimensions(stacked_d_dec, [2, 1, 0, 3])  # dec time x batch x enc time x da
        tanh_add = K.tanh(stacked_d_dec + d_enc)  # dec time x batch x enc time x da
        scores = K.dot(tanh_add, self.W_score)
        if self.use_bias:
            scores = K.bias_add(scores, self.bias_score)
        scores = K.squeeze(scores, 3)  # batch x dec time x enc time

        weights = K.softmax(scores)  # dec time x batch x enc time
        weights = K.expand_dims(weights)

        weighted_encodings = weights * encodings  # dec time x batch x enc time x h
        contexts = K.sum(weighted_encodings, axis=2)  # dec time x batch x h
        contexts = K.permute_dimensions(contexts, [1, 0, 2])  # batch x dec time x h

        return contexts

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list) and len(input_shape) == 2
        assert input_shape[-1]
        output_shape = list(input_shape[1])
        output_shape[-1] = self.units
        return tuple(output_shape)

    def get_config(self):
        config = {
            'units': self.units,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super(Attention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class DenseTransposeTied(Layer):
    @interfaces.legacy_dense_support
    def __init__(self, units,
                 tied_to=None,  # Enter a layer as input to enforce weight-tying
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(DenseTransposeTied, self).__init__(**kwargs)
        self.units = units
        # We add these two properties to save the tied weights
        self.tied_to = tied_to
        self.tied_weights = self.tied_to.weights
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]

        # We remove the weights and bias because we do not want them to be trainable
        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        self.built = True

    def call(self, inputs, **kwargs):
        # Return the transpose layer mapping using the explicit weight matrices
        output = K.dot(inputs, K.transpose(self.tied_weights[0]))
        if self.use_bias:
            output = K.bias_add(output, self.bias, data_format='channels_last')

        if self.activation is not None:
            output = self.activation(output)

        return output

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1]
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)

    def get_config(self):
        config = {
            'units': self.units,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super(DenseTransposeTied, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))