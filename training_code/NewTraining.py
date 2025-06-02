import os
import sys
import re
import datetime
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import random
import shutil
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import kagglehub

# === Save terminal output ===
class DualLogger:
    def __init__(self, file_path):
        self.terminal = sys.stdout
        self.log = open(file_path, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = DualLogger("full_terminal_output.txt")

# === Setup paths ===
log_dir = os.path.join(os.getcwd(), "logs")
results_dir = os.path.join(os.getcwd(), "models")
os.makedirs(log_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)
log_file_path = os.path.join(log_dir, "log.txt")
log_file = open(log_file_path, "w", encoding="utf-8")
def log(msg):
    print(msg)
    log_file.write(msg + "\n")

# === Dataset ===
log(" Checking UTKFace dataset...")
need_download = True
if need_download:
    pathutkface = kagglehub.dataset_download("jangedoo/utkface-new")
    all_images_dir = None
    for root, dirs, files in os.walk(pathutkface):
        if any(f.lower().endswith('.jpg') for f in files):
            all_images_dir = root
            break
    if not all_images_dir:
        raise FileNotFoundError("No .jpg images found after dataset download.")

    all_files = [f for f in os.listdir(all_images_dir) if f.lower().endswith('.jpg')]
    selected_files = random.sample(all_files, len(all_files) // 2)
    temp_folder = os.path.join(pathutkface, "subset")
    os.makedirs(temp_folder, exist_ok=True)

    copied = 0
    for file in selected_files:
        src = os.path.join(all_images_dir, file)
        dst = os.path.join(temp_folder, file)
        if not os.path.exists(dst):
            try:
                shutil.copy(src, dst)
                copied += 1
            except Exception as e:
                print(f"Error copying {file}: {e}")
    log(f" Copied {copied} images to subset folder.")
    pathutkface = temp_folder
else:
    log(f"Using local dataset from: {pathutkface}")

def find_image_dir(root_path):
    for dirpath, _, filenames in os.walk(root_path):
        if any(f.lower().endswith(".jpg") for f in filenames):
            return dirpath
    return None

image_dir = find_image_dir(pathutkface)
if image_dir is None:
    raise FileNotFoundError("No images found.")

file_list = os.listdir(image_dir)
log(f" Total .jpg images in dataset: {len([f for f in file_list if f.lower().endswith('.jpg')])}")
data = []
for filename in file_list:
    match = re.match(r"(\d+)_\d+_\d+_\d+.jpg", filename)
    if match:
        age = int(match.group(1))
        data.append([filename, age])

augmented_data = []
for entry in data:
    augmented_data.append(entry)
    if random.random() < 0.5:
        augmented_data.append(entry)
        augmented_data.append(entry)

df = pd.DataFrame(augmented_data, columns=["filename", "age"])
log(f"Parsed and augmented to {len(df)} samples")

def age_to_class(age):
    if age <= 20: return 0
    elif age <= 40: return 1
    elif age <= 60: return 2
    elif age <= 80: return 3
    else: return 4

df["age_class"] = df["age"].apply(age_to_class)
df = shuffle(df, random_state=42)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# === Data Generators ===
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 64

train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=20,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.15,
    zoom_range=0.25,
    horizontal_flip=True,
    fill_mode='nearest'
)
test_datagen = ImageDataGenerator(rescale=1.0/255)

train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=image_dir,
    x_col='filename',
    y_col='age_class',
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='raw',
    shuffle=True
)
test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    directory=image_dir,
    x_col='filename',
    y_col='age_class',
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='raw',
    shuffle=False
)

# === Model Save Paths ===
model_file_path = os.path.join(results_dir, "trained_age_model_v21.keras")
backup_dir = "D:/BackupModels"
os.makedirs(backup_dir, exist_ok=True)
backup_h5_path = os.path.join(backup_dir, "trained_age_model_v21.h5")

# === Load or Train Model ===
if os.path.exists(model_file_path):
    log(" Loaded model from models folder. Skipping training.")
    model = tf.keras.models.load_model(model_file_path)
else:
    log(" Building and training new model...")
    base_model = VGG16(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    base_model.trainable = True
    for layer in base_model.layers[:-4]:
        layer.trainable = False

    x = Flatten()(base_model.output)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.4)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    output = Dense(5, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=output)
    model.compile(optimizer=Adam(learning_rate=1e-4),
                  loss='sparse_categorical_crossentropy',
                  metrics=['sparse_categorical_accuracy'])

    checkpoint_cb = ModelCheckpoint(model_file_path, save_best_only=True, monitor='val_sparse_categorical_accuracy', mode='max')
    tensorboard_cb = TensorBoard(log_dir=os.path.join(log_dir, "fit_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
    earlystopping = EarlyStopping(monitor="val_loss", mode="min", patience=4, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1, min_lr=1e-6)

    history = model.fit(
    train_generator,
    validation_data=test_generator,
    epochs=25,
    callbacks=[checkpoint_cb, tensorboard_cb, earlystopping, reduce_lr],
    verbose=2
    )


    # === Save Model in Two Places ===
    model.save(model_file_path)      # .keras format for GitHub
    model.save(backup_h5_path)       # .h5 format for local backup
    log(f" Model saved to {model_file_path} and also backed up to {backup_h5_path}")

    # === Plot Results ===
    plt.figure()
    plt.plot(history.history['sparse_categorical_accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_sparse_categorical_accuracy'], label='Val Accuracy')
    plt.title("Accuracy over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, "accuracy_plot.png"))

    plt.figure()
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title("Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, "loss_plot.png"))

    log(" Accuracy and loss plots saved to results directory.")

log(" All done.")
log_file.close()
