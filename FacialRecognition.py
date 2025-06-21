import cv2
import os
import uuid
import numpy as np
import tarfile
from matplotlib import pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow info and warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

try:
    import logging
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D, Input, Flatten, Layer
    from tensorflow.keras.metrics import Precision, Recall
except ImportError as e:
    # print("Error importing TensorFlow or related libraries:", e)
    pass

# Setup paths
DATA_DIR = 'data'
POS_PATH = os.path.join(DATA_DIR, 'positive')
NEG_PATH = os.path.join(DATA_DIR, 'negative')
ANC_PATH = os.path.join(DATA_DIR, 'anchor')
def setup_and_extract_dataset():
    # Make the directories
    os.makedirs(POS_PATH, exist_ok=True)
    os.makedirs(NEG_PATH, exist_ok=True)
    os.makedirs(ANC_PATH, exist_ok=True)

    # Uncompress the TAR GZ labelled Faces in the Wild dataset
    tar_file = 'lfw.tgz'
    if os.path.exists(tar_file):
        with tarfile.open(tar_file) as file:
            file.extractall('lfw')
        # print("Dataset extracted successfully.")
    else:
        pass  # print("Dataset file not found. Please download lfw.tgz from https://vis-www.cs.umass.edu/lfw/")

    # Move lfw data to the 'data/negative' directory
    for directory in os.listdir('lfw'):
        dir_path = os.path.join('lfw', directory)
        if os.path.isdir(dir_path):  # Ensure it's a directory
            for file in os.listdir(dir_path):
                EX_PATH = os.path.join(dir_path, file)
                NEW_PATH = os.path.join(NEG_PATH, file)
                os.replace(EX_PATH, NEW_PATH)
    # print("Dataset organized successfully.")

def collect_images():
    # Establish a connection to the webcam
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()

        # Cut down frame to 250x250px
        frame = frame[120:120 + 250, 200:200 + 250, :]

        # Collect anchors
        if cv2.waitKey(1) & 0xFF == ord('a'):
            imgname = os.path.join(ANC_PATH, '{}.jpg'.format(uuid.uuid1()))
            cv2.imwrite(imgname, frame)
            # print(f"Anchor image saved: {imgname}")

        # Collect positives
        if cv2.waitKey(1) & 0xFF == ord('p'):
            imgname = os.path.join(POS_PATH, '{}.jpg'.format(uuid.uuid1()))
            cv2.imwrite(imgname, frame)
            # print(f"Positive image saved: {imgname}")

        # Show image back to screen
        cv2.imshow('Image Collection', frame)

        # Breaking gracefully
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam
    cap.release()
    # Close the image show frame
    cv2.destroyAllWindows()

# Create datasets
anchor = tf.data.Dataset.list_files(f'{ANC_PATH}/*.jpg').take(300)
positive = tf.data.Dataset.list_files(f'{POS_PATH}/*.jpg').take(300)
negative = tf.data.Dataset.list_files(f'{NEG_PATH}/*.jpg').take(300)

def verify_file_exists(file_path):
    if os.path.exists(file_path):
        pass  # print(f"File exists: {file_path}")
    else:
        pass  # print(f"File does not exist: {file_path}")

# Example usage
verify_file_exists('data\\anchor\\765e78b6-6e08-11ef-9012-c03eba37e5bd.jpg')

# print(f"Anchor Path: {ANC_PATH}")
# print(f"Positive Path: {POS_PATH}")
# print(f"Negative Path: {NEG_PATH}")

# Preprocessing
def preprocess(file_path):
    # Read in image from file path
    byte_img = tf.io.read_file(file_path)
    # Load in the image
    img = tf.io.decode_jpeg(byte_img)

    # Preprocessing steps - resizing the image to be 100x100x3
    img = tf.image.resize(img, (100, 100))
    # Scale image to be between 0 and 1
    img = img / 255.0

    # Return image
    return img

# Preprocess a sample image
# img = preprocess('data\\anchor\\765e78b6-6e08-11ef-9012-c03eba37e5bd.jpg')
# plt.imshow(img)
# plt.axis('off')  # Hide axes for better visualization
# plt.show()

# Create labeled dataset
positives = tf.data.Dataset.zip((anchor, positive, tf.data.Dataset.from_tensor_slices(tf.ones(len(anchor)))))
negatives = tf.data.Dataset.zip((anchor, negative, tf.data.Dataset.from_tensor_slices(tf.zeros(len(anchor)))))

# Combine positives and negatives
data = positives.concatenate(negatives)

# Display an example from the dataset
samples = data.as_numpy_iterator()
example = samples.next()
# print(example)

# Build Train and Test Partition
def preprocess_twin(input_img, validation_img, label):
    return preprocess(input_img), preprocess(validation_img), label

res = preprocess_twin(*example)
# plt.imshow(res[1])

# Build dataloader pipeline
data = data.map(preprocess_twin)
data = data.cache()
data = data.shuffle(buffer_size=1024)
# Training partition
train_data = data.take(round(len(data)*.7))
train_data = train_data.batch(16)
train_data = train_data.prefetch(8)
# Testing partition
test_data = data.skip(round(len(data)*.7))
test_data = test_data.take(round(len(data)*.3))
test_data = test_data.batch(16)
test_data = test_data.prefetch(8)

# Building embedding layer

def make_embedding():
    inp = Input(shape=(100, 100, 3), name='input_image')

    # First block
    c1 = Conv2D(64, (10, 10), activation='relu')(inp)
    m1 = MaxPooling2D((2, 2), padding='same')(c1)

    # Second block
    c2 = Conv2D(128, (7, 7), activation='relu')(m1)
    m2 = MaxPooling2D((2, 2), padding='same')(c2)

    # Third block
    c3 = Conv2D(128, (4, 4), activation='relu')(m2)
    m3 = MaxPooling2D((2, 2), padding='same')(c3)

    # Final embedding block
    c4 = Conv2D(256, (4, 4), activation='relu')(m3)
    f1 = Flatten()(c4)
    d1 = Dense(4096, activation='sigmoid')(f1)

    return Model(inputs=inp, outputs=d1, name='embedding')

embedding = make_embedding()
# embedding.summary()

# Siamese L1 Distance class
class L1Dist(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        # return tf.math.abs(input_embedding - validation_embedding)
        input_embedding, validation_embedding = inputs
        return tf.math.abs(input_embedding - validation_embedding)

l1 = L1Dist()

# Make Siamese model
def make_siamese_model():
    input_image = Input(name='input_img', shape=(100, 100, 3))
    validation_image = Input(name='validation_img', shape=(100, 100, 3))
    inp_embedding = embedding(input_image)
    val_embedding = embedding(validation_image)
    siamese_layer = L1Dist()
    distances = siamese_layer([embedding(input_image), embedding(validation_image)])
    classifier = Dense(1, activation='sigmoid')(distances)
    return Model(inputs=[input_image, validation_image], outputs=classifier, name='SiameseNetwork')

siamese_model = make_siamese_model()
# siamese_model.summary()


#TRAINING MODEL
#Setup loss and optimiser

binary_cross_loss = tf.losses.BinaryCrossentropy()
opt = tf.keras.optimizers.Adam(1e-4) # 0.0001

#Establishing Checkpoints
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
checkpoint = tf.train.Checkpoint(opt=opt, siamese_model=siamese_model)

#Build train step function
test_batch = train_data.as_numpy_iterator()
batch_1 = test_batch.next()
X = batch_1[:2]
y = batch_1[2]


@tf.function
def train_step(batch):
    # Record all of our operations
    with tf.GradientTape() as tape:
        # Get anchor and positive/negative image
        X = batch[:2]
        # Get label
        y = batch[2]

        # Forward pass
        yhat = siamese_model(X, training=True)
        # Calculate loss
        loss = binary_cross_loss(y, yhat)
    print(loss)

    # Calculate gradients
    grad = tape.gradient(loss, siamese_model.trainable_variables)

    # Calculate updated weights and apply to siamese model
    opt.apply_gradients(zip(grad, siamese_model.trainable_variables))

    # Return loss
    return loss

#Build Training Loop
def train(data, EPOCHS):
    # Loop through epochs
    for epoch in range(1, EPOCHS + 1):
        print('\n Epoch {}/{}'.format(epoch, EPOCHS))
        progbar = tf.keras.utils.Progbar(len(data))

        # Loop through each batch
        for idx, batch in enumerate(data):
            # Run train step here
            train_step(batch)
            progbar.update(idx + 1)

        # Save checkpoints
        if epoch % 10 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

#Train the model
#EPOCHS = 50
#train(train_data, EPOCHS)

#Get a batch of test data
test_input, test_val, y_true=  test_data.as_numpy_iterator().next()

#Make Predictions

y_hat = siamese_model.predict([test_input,test_val])

#Post processing the results

[1 if prediction >0.5 else 0 for prediction in y_hat]

y_true

# Creating a metric object
m = Recall()

# Calculating the recall value
m.update_state(y_true, y_hat)

# Return Recall Result
m.result().numpy()

# Creating a metric object
m = Precision()

# Calculating the recall value
m.update_state(y_true, y_hat)

# Return Recall Result
m.result().numpy()

r = Recall()
p = Precision()


#VISUALISE THE RESULTS#

# Set plot size
#plt.figure(figsize=(10,8))

# Set first subplot
#plt.subplot(1,2,1)
#plt.imshow(test_input[3])

# Set second subplot
#plt.subplot(1,2,2)
#plt.imshow(test_val[3])

# Renders cleanly
#plt.show()



#SAVE MODEL#

#Save weights

siamese_model.save('siamesemodel.keras')

# Reload model
siamese_model = tf.keras.models.load_model('siamesemodel.keras',
                                   custom_objects={'L1Dist':L1Dist, 'BinaryCrossentropy':tf.losses.BinaryCrossentropy})

# Make predictions with reloaded model
siamese_model.predict([test_input, test_val])

# View model summary
#siamese_model.summary()

#REAL TIME TESTING

application_data_path = 'application_data'
verification_images_path = os.path.join(application_data_path, 'verification_images')
input_image_path = os.path.join(application_data_path, 'input_image', 'input_image.jpg')

# Example of iterating over the verification images
for image in os.listdir(verification_images_path):
    validation_img = os.path.join(verification_images_path, image)
    #print(validation_img)

## def update():

  ##  global detection_threshold, verification_threshold
  ##  detection_threshold += 0.2
  ##  verification_threshold += 0.2

def verify(model, detection_threshold, verification_threshold):
    # Build results array
    results = []
    for image in os.listdir(os.path.join('application_data', 'verification_images')):
        input_img = preprocess(os.path.join('application_data', 'input_image', 'input_image.jpg'))
        validation_img = preprocess(os.path.join('application_data', 'verification_images', image))

        # Make Predictions
        result = model.predict(list(np.expand_dims([input_img, validation_img], axis=1)))
        results.append(result)

    # Detection Threshold: Metric above which a prediciton is considered positive
    detection = np.sum(np.array(results) > detection_threshold)

    # Verification Threshold: Proportion of positive predictions / total positive samples
    verification = detection / len(os.listdir(os.path.join('application_data', 'verification_images')))
    verified = verification > verification_threshold

    return results, verified

#OPEN CV

detection_threshold = 0.4
verification_threshold = 0.4
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    frame = frame[120:120 + 250, 200:200 + 250, :]
    cv2.imshow('Verification', frame)
    # Verification trigger
    if cv2.waitKey(10) & 0xFF == ord('v'):
        # Save input image to application_data/input_image folder
        cv2.imwrite(os.path.join('application_data', 'input_image', 'input_image.jpg'), frame)
        # Run verification
        results, verified = verify(siamese_model,detection_threshold,verification_threshold)
        print(verified)
        update()
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

np.sum(np.squeeze(results) > 0.6)



# Flags to control which parts of the script run
RUN_SETUP = False
RUN_COLLECTION = False
RUN_PREPROCESS = False
RUN_EMBEDDING = False
RUN_SIAMISE = False
if RUN_SETUP:
    setup_and_extract_dataset()

if RUN_COLLECTION:
    collect_images()

if RUN_PREPROCESS:
    img = preprocess('data\\anchor\\765e78b6-6e08-11ef-9012-c03eba37e5bd.jpg')
    img_numpy = img.numpy()
    print(f"Max value in image: {img_numpy.max()}")
    plt.imshow(img)
    plt.axis('off')
    plt.show()

if RUN_EMBEDDING:
    make_embedding()
    embedding.summary()
if RUN_SIAMISE:
    make_siamese_model()
    siamese_model.summary()