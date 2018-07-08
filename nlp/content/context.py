from keras import backend as K
from keras import activations
from keras import initializers
from keras import regularizers
from keras import constraints
from keras.engine import Layer
from keras.legacy import interfaces
from keras.layers import Input
from keras.layers import Dense
from keras.models import Model


class Content(Layer):
    """

    This layer can only be used as the first layer in a model.

    # Example

    # Arguments
      embeddings_initializer: Initializer for the `embeddings` matrix
          (see [initializers](../initializers.md)).
      embeddings_regularizer: Regularizer function applied to
          the `embeddings` matrix
          (see [regularizer](../regularizers.md)).
      embeddings_constraint: Constraint function applied to
          the `embeddings` matrix
          (see [constraints](../constraints.md)).
      mask_zero: Whether or not the input value 0 is a special "padding"
          value that should be masked out.
          This is useful when using [recurrent layers](recurrent.md)
          which may take variable length input.
          If this is `True` then all subsequent layers
          in the model need to support masking or an exception will be raised.
          If mask_zero is set to True, as a consequence, index 0 cannot be
          used in the vocabulary (input_dim should equal size of
          vocabulary + 1).
      input_length: Length of input sequences, when it is constant.
          This argument is required if you are going to connect
          `Flatten` then `Dense` layers upstream
          (without it, the shape of the dense outputs cannot be computed).

    # Input shape
        2D tensor with shape: `(batch_size, sequence_length)`.

    # Output shape
        3D tensor with shape: `(batch_size, sequence_length, output_dim)`.

    # References
    """

    @interfaces.legacy_embedding_support
    def __init__(self,
                 units,
                 contexts,
                 activation=None,
                 embeddings_initializer='uniform',
                 embeddings_regularizer=None,
                 embeddings_constraint=None,

                 use_bias=False,
                 bias_initializer='zeros',
                 bias_regularizer=None,
                 bias_constraint=None,

                 input_length=None,

                 **kwargs):

        if 'input_shape' not in kwargs:
            if input_length:
                kwargs['input_shape'] = (input_length,)
            else:
                kwargs['input_shape'] = (None,)
        super(Content, self).__init__(**kwargs)

        self.units = units
        self.contexts = contexts
        self.activation = activations.get(activation)
        self.embeddings_initializer = initializers.get(embeddings_initializer)
        self.embeddings_regularizer = regularizers.get(embeddings_regularizer)
        self.use_bias = use_bias
        self.bias_initializer = initializers.get(bias_initializer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.bias_constraint = constraints.get(bias_constraint)
        self.embeddings_constraint = constraints.get(embeddings_constraint)
        self.input_length = input_length

        self.embeddings = None
        self.bias = None

        self.built = False

    def build(self, input_shape):
        assert input_shape and len(input_shape) == 2 and len(input_shape[1]) >= 2
        assert input_shape[1][-1]
        input_dim = input_shape[1][-1]
        self.embeddings = self.add_weight(
            shape=(self.contexts, input_dim, self.units),
            initializer=self.embeddings_initializer,
            name='kernel',
            regularizer=self.embeddings_regularizer,
            constraint=self.embeddings_constraint,
            dtype=self.dtype)

        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        self.built = True

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2 and len(input_shape[1]) >= 2
        assert input_shape[1][-1]
        output_shape = list(input_shape[1])
        output_shape[-1] = self.units
        return tuple(output_shape)

    def call(self, inputs, **kwargs):
        """

        :param inputs: [content_idx, inputs]
        :param kwargs:
        :return:
        """
        content_idx = inputs[0]
        inputs = inputs[1]
        if K.dtype(content_idx) != 'int32':
            content_idx = K.cast(content_idx, 'int32')

        # 从embeddings取出对应的权重
        w = K.gather(self.embeddings, content_idx)

        output = K.dot(inputs, w)

        if self.use_bias:
            output = K.bias_add(output, self.bias)

        if self.activation is not None:
            output = self.activation(output)

        return output

    def get_config(self):
        config = {'units': self.units,
                  'embeddings_initializer': initializers.serialize(self.embeddings_initializer),
                  'embeddings_regularizer': regularizers.serialize(self.embeddings_regularizer),
                  'embeddings_constraint': constraints.serialize(self.embeddings_constraint),
                  'input_length': self.input_length}
        base_config = super(Content, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


if __name__ == '__main__':
    """
    DEMO
    """

    x = Input(shape=(128,), dtype='float32', name='x')
    context_id = Input(shape=(1,), dtype='int32', name='context_id')

    l_context = Content(
        units=10
        , contexts=10
        , activation='sigmoid'
        , embeddings_initializer='lecun_uniform'
    )

    o_context = l_context([context_id, x])

    y = Dense(units=1, activation='softmax', name='y')(o_context)

    model = Model(inputs=[context_id, x], outputs=[y])

    model.summary()


    model.compile(
        optimizer='adam',
        loss={'y': 'binary_crossentropy'},
        loss_weights={'y': 1.},
        metrics=['accuracy']
    )



    # model.fit(
    #     x={'x_int': x_int, 'x_float': x_float},
    #     y={'y': y, 'y_aux': y},
    #     class_weight=class_weight,
    #     epochs=50, batch_size=64, shuffle=True, validation_split=0.2,
    #     callbacks=[early_stopping, checkpoint],
    #     verbose=1,
    # )