# Facial Recognition with Siamese Network

This project implements a facial recognition system using a Siamese neural network in TensorFlow and Keras.

## Features

- Collects anchor and positive images using your webcam.
- Uses the Labeled Faces in the Wild (LFW) dataset for negative samples.
- Preprocesses images and builds a custom embedding model.
- Trains a Siamese network to verify if two faces are of the same person.
- Includes real-time verification using your webcam.

## Requirements

- Python 3.7+
- TensorFlow 2.x
- OpenCV
- NumPy
- Matplotlib

Install dependencies with:

```bash
pip install tensorflow opencv-python numpy matplotlib
```

## Usage

1. **Setup the dataset**  
   Download the [LFW dataset](https://vis-www.cs.umass.edu/lfw/lfw.tgz) and place `lfw.tgz` in your project directory.  
   Uncomment and run:
   ```python
   RUN_SETUP = True
   ```

2. **Collect images**  
   Uncomment and run:
   ```python
   RUN_COLLECTION = True
   ```
   Use your webcam to collect anchor (`a`) and positive (`p`) images.

3. **Train the model**  
   Uncomment and set the number of epochs:
   ```python
   # EPOCHS = 50
   # train(train_data, EPOCHS)
   ```

4. **Real-time verification**  
   The script supports real-time face verification using your webcam. Press `v` to verify, `q` to quit.

## Note

- The trained model file (`siamesemodel.keras`) is **not included** due to GitHub file size limits.  
- To use your own trained model, save it as `siamesemodel.keras` in the project directory.

## License

MIT License
