import tensorflow as tf
from tensorflow.keras import layers, models, applications

def convolution_block(block_input, num_filters=256, kernel_size=3, dilation_rate=1, padding="same", use_bias=False):
    x = layers.Conv2D(
        num_filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding=padding,
        use_bias=use_bias,
        kernel_initializer=tf.keras.initializers.HeNormal(),
    )(block_input)
    x = layers.BatchNormalization()(x)
    return layers.Activation("relu")(x)

def DilatedSpatialPyramidPooling(dspp_input):
    # global average pooling
    x = layers.GlobalAveragePooling2D()(dspp_input)
    x = layers.Reshape((1, 1, x.shape[-1]))(x)
    # 1x1 convolution with 256 filters
    x = convolution_block(x, num_filters=256, kernel_size=1, use_bias=True)
    # bilinearly upsample features
    out_pool = layers.Lambda(lambda tensors: tf.image.resize(tensors[0], tf.shape(tensors[1])[1:3]))([x, dspp_input])

    # one 1x1 convolution and three 3x3 convolutions pre-activation
    out_1 = convolution_block(dspp_input, num_filters=256, kernel_size=1, dilation_rate=1)
    out_6 = convolution_block(dspp_input, num_filters=256, kernel_size=3, dilation_rate=6)
    out_12 = convolution_block(dspp_input, num_filters=256, kernel_size=3, dilation_rate=12)
    out_18 = convolution_block(dspp_input, num_filters=256, kernel_size=3, dilation_rate=18)

    x = layers.Concatenate(axis=-1)([out_1, out_6, out_12, out_18, out_pool])
    output = convolution_block(x, num_filters=256, kernel_size=1)
    return output

def build_deeplabv3_plus(input_shape, num_classes=8):
    model_input = layers.Input(shape=input_shape)

    # 1. Backbone: ResNet50V2
    resnet50v2 = applications.ResNet50V2(
        weights=None, include_top=False, input_tensor=model_input
    )

    # Extraction des features
    x = resnet50v2.get_layer("post_relu").output if "post_relu" in [layer.name for layer in resnet50v2.layers] else resnet50v2.get_layer("conv4_block6_out").output
    input_a = resnet50v2.get_layer("conv2_block3_1_relu").output if "conv2_block3_1_relu" in [layer.name for layer in resnet50v2.layers] else resnet50v2.get_layer("conv2_block3_out").output

    # 2. ASPP
    x = DilatedSpatialPyramidPooling(x)

    # 3. Decoder
    input_a = convolution_block(input_a, num_filters=48, kernel_size=1)

    x = layers.Lambda(lambda tensors: tf.image.resize(tensors[0], tf.shape(tensors[1])[1:3]))([x, input_a])
    x = layers.Concatenate(axis=-1)([x, input_a])

    x = convolution_block(x, num_filters=256)
    x = convolution_block(x, num_filters=256)

    x = layers.Lambda(lambda tensors: tf.image.resize(tensors[0], tf.shape(tensors[1])[1:3]))([x, model_input])
    model_output = layers.Conv2D(num_classes, kernel_size=(1, 1), padding="same", activation="softmax")(x)

    model = models.Model(inputs=model_input, outputs=model_output, name="DeepLabV3_ResNet50V2_Official")

    return model
