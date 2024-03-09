# TuneTrend: Decade & Century Song Classification

## Overview
The Song Decade and Century Classifier is a machine-learning project designed to classify songs based on the decade or century in which they were released. The classifier utilizes various features of songs, such as audio characteristics, lyrics, and metadata, to accurately predict the time period to which each song belongs.

## Features
- **Data Collection**: The project includes scripts to collect song data from sources like Spotify, music databases, or CSV files. Data collection is customizable, allowing users to specify criteria such as genre, artist, or release year.

- **Feature Extraction**: Once the data is collected, features are extracted from each song. This includes audio features extracted using libraries like Librosa, textual features extracted from song lyrics, and metadata features such as artist information, album name, and release date.

- **Machine Learning Classification**: The extracted features are used to train machine learning models for classification. The project supports classification into decades (e.g., 1960s, 1970s) or centuries (e.g., 19th century, 20th century). Classification models may include Random Forest, Support Vector Machines, or neural networks.

- **Evaluation and Metrics**: The trained models are evaluated using various metrics such as accuracy, precision, recall, and F1 score. Evaluation results are provided to assess the performance of the classification models.

## Usage
1. **Data Collection**: Run the data collection script to gather song data from the desired source. Customize the script to specify criteria such as genre, artist, or release year.
   ```
   python collect_data.py
   ```

2. **Preprocessing and Feature Extraction**: Preprocess the collected data and extract features using the preprocessing script.
   ```
   python preprocess_data.py
   ```

3. **Model Training**: Train the classification model using the training script. Choose the desired classification task (decades or centuries) and the machine learning algorithm.
   ```
   python train_model.py --task decades --algorithm RandomForest
   ```

4. **Model Evaluation**: Evaluate the trained model and view accuracy metrics.
   ```
   python evaluate_model.py
   ```
   
## License
This project is licensed under the MIT License. See the `LICENSE` file for details.
