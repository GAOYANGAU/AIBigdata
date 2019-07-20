"""
使用keras在VGG16网络的基础上实现FCN32
"""
from keras.applications import vgg16
from keras.models import Model
from keras.layers import Conv2D, Conv2DTranspose, Input, Dropout, Reshape, Activation
from keras.utils import plot_model


def fcn32(n_classes, input_height, input_width):

    assert input_height % 32 == 0
    assert input_width % 32 == 0

    img_input = Input(shape=(input_height, input_width, 3))

    model = vgg16.VGG16(
        include_top=False,
        weights='imagenet', input_tensor=img_input)
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

    x = Conv2DTranspose(filters=n_classes, kernel_size=(32, 32), strides=(32, 32), padding="valid", activation=None,
                        name="score2")(x)

    x = Reshape((-1, n_classes))(x)
    x = Activation("softmax")(x)

    fcn8 = Model(inputs=img_input, outputs=x)
    # mymodel.summary()
    return fcn8


if __name__ == '__main__':
    m = fcn32(15, 320, 320)
    m.summary()
    plot_model(m, show_shapes=True, to_file='model_fcn32.png')
    print(len(m.layers))
