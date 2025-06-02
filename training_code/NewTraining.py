import os
import datetime
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, ReduceLROnPlateau, Callback
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import kagglehub

# === Setup paths ===
log_dir = os.path.join(os.getcwd(), "logs")
results_dir = os.path.join(os.getcwd(), "models")
os.makedirs(log_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)
log_file_path = os.path.join(log_dir, "log.txt")

def log(msg):
    print(msg)
    with open(log_file_path, "a", encoding="utf-8") as log_file:
        log_file.write(msg + "\n")

class EpochLogger(Callback):
    def on_epoch_end(self, epoch, logs=None):
        log(f"Epoch {epoch+1}: loss={logs['loss']:.4f}, acc={logs['accuracy']:.4f}, val_loss={logs['val_loss']:.4f}, val_acc={logs['val_accuracy']:.4f}")

def map_age_to_group(age):
    if age <= 18:
        return 0
    elif age <= 30:
        return 1
    elif age <= 50:
        return 2
    elif age <= 70:
        return 3
    else:
        return 4

AGE_GROUPS = {
    0: "0-18",
    1: "19-30",
    2: "31-50",
    3: "51-70",
    4: "71+"
}

# === Load dataset ===
dataset_dir = os.path.join(os.getcwd(), "Images")
if not os.path.exists(dataset_dir):
    dataset_dir = kagglehub.dataset_download("jangedoo/utkface-new")

image_paths = []
if os.path.exists(dataset_dir):
    image_paths = [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir) if f.lower().endswith(".jpg")]

if not image_paths:
    raise FileNotFoundError(f"No .jpg images found in: {dataset_dir}")

ages = [int(os.path.basename(path).split("_")[0]) for path in image_paths]
age_groups = [map_age_to_group(age) for age in ages]

df = pd.DataFrame({
    "path": image_paths,
    "label": [str(g) for g in age_groups]
})

train_df, val_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)

train_gen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=15,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    brightness_range=(0.8, 1.2)
)

val_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_data = train_gen.flow_from_dataframe(
    train_df, x_col="path", y_col="label", target_size=(224, 224),
    batch_size=32, class_mode="sparse"
)

val_data = val_gen.flow_from_dataframe(
    val_df, x_col="path", y_col="label", target_size=(224, 224),
    batch_size=32, class_mode="sparse"
)

# === Compute class weights ===
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=df["label"].unique(),
    y=df["label"]
)
class_weights = dict(enumerate(class_weights))

# === Model definition ===
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

x = Flatten()(base_model.output)
x = Dense(512, activation='relu')(x)
x = Dropout(0.4)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.3)(x)
output = Dense(5, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=output)

model.compile(optimizer=Adam(1e-4), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
model_path = os.path.join(results_dir, f"age_model_vgg16_{timestamp}.h5")

callbacks = [
    EpochLogger(),
    ModelCheckpoint(model_path, monitor='val_accuracy', save_best_only=True),
    TensorBoard(log_dir=os.path.join(log_dir, f"tb_{timestamp}")),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1),
    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
]

# === Train ===
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=20,
    callbacks=callbacks,
    class_weight=class_weights,
    verbose=2
)

# === Plot metrics ===
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Accuracy over Epochs')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss over Epochs')
plt.legend()
plt.show()
