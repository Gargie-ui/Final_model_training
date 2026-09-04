import tensorflow as tf
from src import config

def build_full_model(unfreeze_top_n=0):
    """
    Builds the complete model: MobileNetV2 backbone + custom classification head.
    
    MobileNetV2 has ~3.4 million parameters (vs DenseNet121's ~7M+), making full
    end-to-end backpropagation on CPU significantly faster and practical.
    """
    # 1. Load the pre-trained MobileNetV2 backbone
    # include_top=False excludes the original 1000-class ImageNet classification head.
    base_model = tf.keras.applications.MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(config.IMG_SIZE, config.IMG_SIZE, 3)
    )
    
    # 2. Set backbone to trainable for full end-to-end fine-tuning
    base_model.trainable = True
    print("MobileNetV2 backbone set to TRAINABLE for end-to-end fine-tuning.")
    
    # 3. Define the custom classification head using Keras Functional API
    inputs = tf.keras.Input(shape=(config.IMG_SIZE, config.IMG_SIZE, 3), name="input_image")
    
    # Forward pass through backbone in inference mode (keeps batch norm statistics stable)
    x = base_model(inputs, training=False)
    
    # Global average pooling reduces feature map shape from (7, 7, 1280) to (1280,)
    x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
    
    # Dropout layer for regularization
    x = tf.keras.layers.Dropout(0.3, name="head_dropout", seed=config.RANDOM_SEED)(x)
    
    # Softmax output layer for the 4 target classes
    outputs = tf.keras.layers.Dense(
        units=len(config.CLASSES),
        activation="softmax",
        name="dense_output"
    )(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="mobilenet_v2_classifier")
    
    return model

def build_head_model():
    """
    Builds a standalone head model for 1280-dimensional MobileNetV2 bottleneck features.
    """
    inputs = tf.keras.Input(shape=(1280,), name="bottleneck_features")
    x = tf.keras.layers.Dropout(0.3, name="head_dropout", seed=config.RANDOM_SEED)(inputs)
    outputs = tf.keras.layers.Dense(
        units=len(config.CLASSES),
        activation="softmax",
        name="dense_output"
    )(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="classifier_head_only")
    return model

def sync_head_weights(head_model, full_model):
    """
    Synchronizes trained custom head weights back into the full model.
    """
    print("\nSynchronizing trained head weights to the full MobileNetV2 model...")
    layers_to_sync = ["dense_output"]
    
    for name in layers_to_sync:
        try:
            h_layer = head_model.get_layer(name)
            f_layer = full_model.get_layer(name)
            f_layer.set_weights(h_layer.get_weights())
            print(f"  Successfully synchronized layer: {name}")
        except Exception as e:
            print(f"  Warning: Could not sync weights for layer {name}: {e}")
            
    print("Head weights synchronization complete.")

if __name__ == "__main__":
    full_m = build_full_model()
    full_m.summary()
