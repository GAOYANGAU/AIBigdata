"""
用keras在VVG16网络的基础上实现FCN8
"""
from keras.applications import vgg16
from keras.models import Model
from keras.layers import Conv2D, Conv2DTranspose, Input, add, Dropout, Reshape, Activation


def fcn8_helper(n_classes, input_height, input_width):

    assert input_height % 32 == 0
    assert input_width % 32 == 0

    img_input = Input(shape=(input_height, input_width, 3))

    model = vgg16.VGG16(
        include_top=False,
        weights='data/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5', input_tensor=img_input,
        pooling=None,
        classes=1000)
    assert isinstance(model, Model)

    x = Conv2D(
        filters=4096,
        kernel_size=(
            7,
            7),
        padding="same",
        activation="relu",
        name="fc6")(
            model.output)
    x = Dropout(rate=0.5)(x)
    x = Conv2D(
        filters=4096,
        kernel_size=(
            1,
            1),
        padding="same",
        activation="relu",
        name="fc7")(x)
    x = Dropout(rate=0.5)(x)

    x = Conv2D(filters=n_classes, kernel_size=(1, 1), padding="same", activation="relu", kernel_initializer="he_normal",
               name="score_fr")(x)

    x = Conv2DTranspose(filters=n_classes, kernel_size=(2, 2), strides=(2, 2), padding="valid", activation=None,
                        name="score2")(x)
    fcn = Model(inputs=img_input, outputs=x)
    return fcn


def fcn8_model(n_classes, input_height, input_width):

    fcn8 = fcn8_helper(n_classes, input_height, input_width)

    skip_con1 = Conv2D(n_classes, kernel_size=(1, 1), padding="same", activation=None, kernel_initializer="he_normal",
                       name="score_pool4")(fcn8.get_layer("block4_pool").output)
    summed = add(inputs=[skip_con1, fcn8.output])

    x = Conv2DTranspose(n_classes, kernel_size=(2, 2), strides=(2, 2), padding="valid", activation=None,
                        name="score4")(summed)

    skip_con2 = Conv2D(n_classes, kernel_size=(1, 1), padding="same", activation=None, kernel_initializer="he_normal",
                       name="score_pool3")(fcn8.get_layer("block3_pool").output)
    summed2 = add(inputs=[skip_con2, x])

    up = Conv2DTranspose(n_classes, kernel_size=(8, 8), strides=(8, 8),
                         padding="valid", activation=None, name="upsample")(summed2)
    up = Reshape((-1, n_classes))(up)
    up = Activation("softmax")(up)
    fcn_model = Model(inputs=fcn8.input, outputs=up)

    return fcn_model


if __name__ == '__main__':
    m = fcn8_model(15, 320, 320)
    from keras.utils import plot_model
    plot_model(m, show_shapes=True, to_file='model_fcn8.png')
    print(len(m.layers))
