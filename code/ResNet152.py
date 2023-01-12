import tensorflow as tf

# Load ResNet152 model with pre-trained weights
model = tf.keras.applications.ResNet152(weights='imagenet', include_top=False)

# Freeze the layers of the model to prevent their weights from being updated during training
for layer in model.layers:
  layer.trainable = False

# Add a new classifier on top of the model
x = model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(1024, activation='relu')(x)
predictions = tf.keras.layers.Dense(3, activation='softmax')(x)
model = tf.keras.Model(inputs=model.input, outputs=predictions)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Load the training data from the local directory
image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
train_data_gen = image_generator.flow_from_directory(directory='train',
                                                     target_size=(224, 224),
                                                     batch_size=64,
                                                     class_mode='categorical')
val_data_gen = image_generator.flow_from_directory(directory='val',
                                                   target_size=(224, 224),
                                                   batch_size=64,
                                                   class_mode='categorical')

# Use a data generator to create augmented training data
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)

# Use the data generator to create augmented training data
datagen.fit(train_data_gen)

# Train the model
model.fit_generator(datagen.flow(train_data_gen, batch_size=64),
                    epochs=10,
                    validation_data=val_data_gen)

# Evaluate the model on the test set
_, top1_accuracy = model.evaluate_generator(val_data_gen)

# Create a function to calculate top-2 accuracy
def top_2_accuracy(y_true, y_pred):
    return tf.keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k=2)

# Compile the model with the top-2 accuracy metric
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[top_2_accuracy])

# Evaluate the model on the test set
_, top2_accuracy = model.evaluate_generator(val_data_gen)

print('Top-1 accuracy:', top1_accuracy)

print('Top-2 accuracy:', top2_accuracy)