import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.regularizers import l2
from keras.preprocessing.image import ImageDataGenerator

def create_vgg(num_blocks = 3,
               block_sizes = [2,2,2],
               filter_sizes = [32, 64, 128],
               kernel_sizes = [3,3,3],
               kernel_strides = [1,1,1],
               pool_sizes = [2,2,2],
               pool_strides = [2,2,2],
               pool_dropout_probas = [0.0, 0.0, 0.0],
               num_dense = 1,
               dense_sizes = [256],
               dense_dropout_probas = [0.5]):
    """
    Creates VGG like network of structure

    {
        {
            Conv2D
            Conv2D
            .
            .
            .
            Conv2D
        } x block_sizes[i]
        MaxPooling2D
        Dropout
    } x num_blocks

    {
        Dense
        Dropout
    } x num_dense


    Args:
    num_blocks -- Number of conv--pool blocks.
    block_sizes -- Number of conv layers in ith block.
    filter_sizes -- Size of filters in conv layers in ith block.
    kernel_sizes -- Size of kernel in conv layers in ith block.
    kernel_strides -- Size of stride in conv layers in ith block.
    pool_sizes -- Size of pooling window in ith block.
    pool_strides -- Size of stride of pooling window in ith block.
    pool_dropout_probas -- Dropout probability after the pool layer.
    num_dense -- Number of dense layers.
    dense_sizes -- Sizes of dense layers.
    dense_dropout_probas -- Dropout probas in the dense layers.

    Returns:
    model -- A VGG like keras model.
    """
    model = Sequential()

    for i in range(num_blocks):
        for j in range(block_sizes[i]):
            filters = filter_sizes[i]
            kernel_size = (kernel_sizes[i], kernel_sizes[i])
            strides = (kernel_strides[i], kernel_strides[i])
            if i==0 and j==0:
                model.add(Conv2D(filters = filters,
                                 kernel_size = kernel_size,
                                 strides = strides,
                                 activation="relu",
                                 input_shape=(75,75,2)))
            else:
                model.add(Conv2D(filters = filters,
                                 kernel_size = kernel_size,
                                 strides = strides,
                                 activation="relu"))

        pool_size = (pool_sizes[i], pool_sizes[i])
        pool_stride = (pool_strides[i], pool_strides[i])
        model.add(MaxPooling2D(pool_size,
                               pool_stride))
        rate = pool_dropout_probas[i]
        model.add(Dropout(rate))

    model.add(Flatten())
    for i in range(num_dense):
        model.add(Dense(dense_sizes[i],
                        activation = "relu"))
        model.add(Dropout(dense_dropout_probas[i]))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(optimizer='adam', loss='binary_crossentropy')

    return model


def create_vgg_simple(block_sizes = [2,2,2],
                      filter_sizes = [32, 64, 128],
                      dense_sizes = [256],
                      dense_dropout_probas = [0.5]):
    """
    Creates vgg like model
    simplified call
    """
    num_blocks = len(block_sizes)
    num_dense = len(dense_sizes)
    model = create_vgg(
               num_blocks = num_blocks,
               block_sizes = block_sizes,
               filter_sizes = filter_sizes,
               kernel_sizes = [3,3,3],
               kernel_strides = [1,1,1],
               pool_sizes = [2,2,2],
               pool_strides = [2,2,2],
               pool_dropout_probas = [0.0, 0.0, 0.0],
               num_dense = num_dense,
               dense_sizes = dense_sizes,
               dense_dropout_probas = dense_dropout_probas)

    return model
