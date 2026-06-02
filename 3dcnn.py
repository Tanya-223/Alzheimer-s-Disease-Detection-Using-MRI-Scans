from oasis_3d_data_generator import get_3d_generators
from tensorflow.keras import layers, models

train_gen, val_gen, test_gen = get_3d_generators()

# Build a 3D CNN
model = models.Sequential([
    layers.Conv3D(32, (3,3,3), activation='relu', input_shape=(128,128,16,1)),
    layers.MaxPooling3D((2,2,2)),
    layers.Conv3D(64, (3,3,3), activation='relu'),
    layers.MaxPooling3D((2,2,2)),
    layers.Conv3D(128, (3,3,3), activation='relu'),
    layers.GlobalAveragePooling3D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(4, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

model.fit(train_gen, validation_data=val_gen, epochs=20)
model.evaluate(test_gen)
