# Movie Genre Classifier

## Overview
This project implements a convolutional neural network (CNN) using TensorFlow to classify movie posters into multiple genres based on a dataset from Kaggle. The model processes images and predicts genres such as Drama, Comedy, Adventure, etc., using a multi-label classification approach.

## Dataset
The dataset used is the "Movie Classifier" dataset available on Kaggle (`raman77768/movie-classifier`). It includes:
- A CSV file (`train.csv`) containing image IDs and their corresponding genre labels (multi-label binary format).
- A directory of movie poster images (`Images/`).

For this project, the dataset is limited to the first 1,000 entries to reduce computational requirements.

## Requirements
To run the project, install the following Python packages:
- `tensorflow`
- `kagglehub`
- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`
- `tqdm`

You can install them using:
```bash
pip install tensorflow kagglehub numpy pandas matplotlib scikit-learn tqdm
```

## Project Structure
- **OS_project_real_final.ipynb**: The main Jupyter Notebook containing the code for:
  - Downloading and preprocessing the dataset.
  - Loading and processing images using multi-threading.
  - Building and training a CNN model.
  - Evaluating the model and predicting genres for sample images.
- **README.md**: This file, providing an overview and instructions for the project.

## How It Works
1. **Data Preparation**:
   - The dataset is downloaded using `kagglehub`.
   - The CSV file is loaded, and image IDs are modified to include the `.jpg` extension.
   - Images are loaded in parallel using `ThreadPoolExecutor` for efficiency and resized to 200x200 pixels.
   - The dataset is split into training (90%) and validation (10%) sets.

2. **Model Architecture**:
   - A Sequential CNN model is built with:
     - Three convolutional layers (16, 32, and 64 filters) with ReLU activation.
     - Batch normalization, max pooling, and dropout for regularization.
     - A flattened layer followed by dense layers (128 units and 25 output units with sigmoid activation for multi-label classification).
   - The model is compiled with the Adam optimizer and binary cross-entropy loss.

3. **Training**:
   - The model is trained for 15 epochs with a batch size of 32.
   - Training and validation accuracy/loss are plotted to evaluate performance.

4. **Prediction**:
   - The model predicts genre probabilities for a given image (e.g., `tt0086429.jpg`).
   - The top 3 predicted genres are displayed with their probabilities.
   - For a specified image (e.g., `tt0086423.jpg`), the true genres from the CSV are extracted and listed.

## Usage
1. **Setup**:
   - Ensure you have a Kaggle account and API key configured for `kagglehub` to download the dataset.
   - Run the notebook in an environment with GPU support (e.g., Google Colab with T4 GPU) for faster training.

2. **Running the Notebook**:
   - Open `OS_project_real_final.ipynb` in Jupyter Notebook or Google Colab.
   - Execute the cells sequentially to:
     - Install dependencies.
     - Download and preprocess the dataset.
     - Train the model.
     - View predictions and true genres for sample images.

3. **Custom Predictions**:
   - Modify the `image_id` variable in the last cell to check true genres for any image in the dataset.
   - To predict genres for a new image, ensure it is in the correct format (200x200 pixels) and update the code to load and predict using the trained model.

## Results
- The model achieves increasing accuracy over 15 epochs, with validation accuracy reaching approximately 17% and validation loss decreasing to 0.33.
- Example output for `tt0086429.jpg`:
  - Top 3 predicted genres: Drama (0.621), Adventure (0.156), Comedy (0.135).
- The notebook also checks true genres for `tt0086423.jpg` from the CSV, if available.

## Limitations
- The dataset is limited to 1,000 images, which may not capture the full diversity of movie posters.
- The model's validation accuracy is relatively low, indicating potential overfitting or the need for more data or hyperparameter tuning.
- Multi-label classification is challenging due to imbalanced genre distributions.

## Future Improvements
- Increase the dataset size for better generalization.
- Experiment with deeper architectures or pre-trained models (e.g., ResNet, VGG).
- Apply data augmentation to improve robustness.
- Fine-tune hyperparameters (e.g., learning rate, dropout rates).

## License
This project is for educational purposes and uses the Kaggle dataset under its respective license. Ensure compliance with Kaggle's terms when using the dataset.

## Acknowledgments
- Dataset provided by `raman77768` on Kaggle.
- Built with TensorFlow and other open-source Python libraries.
