
# Handwritten Letters Recognition

This project aims to build a Convolutional Neural Network (CNN) to recognize handwritten letters using the EMNIST dataset. The model is implemented using TensorFlow and Keras.

## Project Structure

- `code.ipynb`: Jupyter notebook containing the model implementation and training process.
- `Best_Model.keras`: Saved model with the best validation accuracy.

## Dataset

The dataset used is the EMNIST Letters dataset, which can be found [here](https://www.nist.gov/itl/products-and-services/emnist-dataset).

- `emnist-letters-train.csv`: Training dataset
- `emnist-letters-test.csv`: Testing dataset

## Requirements

Ensure you have the following libraries installed:

- pandas
- numpy
- matplotlib
- seaborn
- tensorflow
- keras
- scikit-learn

You can install the required libraries using:

```bash
pip install pandas numpy matplotlib seaborn tensorflow keras scikit-learn
```

## Data Preparation

The data is loaded and split into training and testing sets. It is normalized to have values between 0 and 1.

## Model Architecture

The model is a Convolutional Neural Network (CNN) with the following layers:

- Conv2D
- BatchNormalization
- MaxPooling2D
- Dropout
- Flatten
- Dense

## Training the Model

The model is trained using the following callbacks to improve performance and manage the training process:

### EarlyStopping

Stops training when the validation accuracy stops improving.

### ReduceLROnPlateau

Reduces the learning rate when the validation loss stops improving.

### ModelCheckpoint

Saves the model with the best validation accuracy.



## Results

The model's performance is tracked using accuracy and loss metrics on both training and validation sets.

## License

This project is licensed under the MIT License.

## Acknowledgments

- TensorFlow and Keras for providing the deep learning framework.
- The EMNIST dataset creators for providing the dataset.

## Contact

For any questions, please contact Lyes HADJAR at lyes.hadjar@ensia.edu.dz
