from tensorflow_model_optimization.python.core.keras.compat import keras #for Quantization Aware Training (QAT)
import tensorflow_model_optimization as tfmot #for Post Training Quantization (PTQ)
from datasets import load_dataset #for downloading the Wake Vision Dataset
import tensorflow as tf #for designing and training the model 

model_name = 'wv_k_8_c_5'

#some hyperparameters 
#Play with them!
input_shape = (50,50,3)
batch_size = 512
learning_rate = 0.001
epochs = 1

#model architecture (with Quantization Aware Training - QAT)
#Play with it!
inputs = keras.Input(shape=input_shape)
#
x = keras.layers.Conv2D(8, (3,3), padding='same')(inputs)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.ReLU()(x)
#
x = keras.layers.MaxPooling2D((2,2))(x)
x = keras.layers.Conv2D(16, (3,3), padding='same')(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.ReLU()(x)
#
x = keras.layers.MaxPooling2D((2,2))(x)
x = keras.layers.Conv2D(24, (3,3), padding='same')(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.ReLU()(x)
#
x = keras.layers.MaxPooling2D((2,2))(x)
x = keras.layers.Conv2D(30, (3,3), padding='same')(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.ReLU()(x)
#
x = keras.layers.MaxPooling2D((2,2))(x)
x = keras.layers.Conv2D(34, (3,3), padding='same')(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.ReLU()(x)
#
x = keras.layers.MaxPooling2D((2,2))(x)
x = keras.layers.Conv2D(37, (3,3), padding='same')(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.ReLU()(x)
#
x = keras.layers.GlobalAveragePooling2D()(x)
#
x = keras.layers.Dense(37)(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.ReLU()(x)
#
outputs = keras.layers.Dense(2)(x)

model = keras.Model(inputs, outputs)

#compile model
opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)

#load dataset
ds = load_dataset("Harvard-Edge/Wake-Vision")
    
train_ds = ds['train_quality'].to_tf_dataset(columns='image', label_cols='person')
val_ds = ds['validation'].to_tf_dataset(columns='image', label_cols='person')
test_ds = ds['test'].to_tf_dataset(columns='image', label_cols='person')

#some preprocessing 
data_preprocessing = tf.keras.Sequential([
    #resize images to desired input shape
    tf.keras.layers.Resizing(input_shape[0], input_shape[1])])

data_augmentation = tf.keras.Sequential([
    data_preprocessing,
    #add some data augmentation 
    #Play with it!
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.2)])
    
train_ds = train_ds.shuffle(1000).map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.map(lambda x, y: (data_preprocessing(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.map(lambda x, y: (data_preprocessing(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE).batch(1).prefetch(tf.data.AUTOTUNE)

#set validation based early stopping
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath= model_name + ".tf",
    monitor='val_sparse_categorical_accuracy',
    mode='max', save_best_only=True)
    
#training
model.fit(train_ds, steps_per_epoch=1, epochs=epochs, validation_data=val_ds, callbacks=[model_checkpoint_callback])

#Post Training Quantization (PTQ)
model = tf.keras.models.load_model(model_name + ".tf")

def representative_dataset():
    for data in train_ds.rebatch(1).take(150) :
        yield [tf.dtypes.cast(data[0], tf.float32)]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8 
converter.inference_output_type = tf.uint8
tflite_quant_model = converter.convert()

with open(model_name + ".tflite", 'wb') as f:
    f.write(tflite_quant_model)
    
#Test quantized model
interpreter = tf.lite.Interpreter(model_name + ".tflite")
interpreter.allocate_tensors()

output = interpreter.get_output_details()[0]  # Model has single output.
input = interpreter.get_input_details()[0]  # Model has single input.

correct = 0
wrong = 0

for image, label in test_ds :
    # Check if the input type is quantized, then rescale input data to uint8
    if input['dtype'] == tf.uint8:
       input_scale, input_zero_point = input["quantization"]
       image = image / input_scale + input_zero_point
       input_data = tf.dtypes.cast(image, tf.uint8)
       interpreter.set_tensor(input['index'], input_data)
       interpreter.invoke()
       if label.numpy() == interpreter.get_tensor(output['index']).argmax() :
           correct = correct + 1
       else :
           wrong = wrong + 1
print(f"\n\nTflite model test accuracy: {correct/(correct+wrong)}\n\n")
