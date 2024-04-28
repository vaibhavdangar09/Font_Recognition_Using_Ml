# Font_Recognition_Using_Ml

![image](https://github.com/vaibhavdangar09/Font_Recognition_Using_Ml/assets/85430510/2e2c5115-f394-4981-b2d6-668dd41e051e)


# Steps to run file:
1. Run Utils.py file
2. Run training file
3. Run test file 

The Font Recognition project employs a combination of Convolutional Neural Networks (CNNs) and Long Short-Term Memory Recurrent Neural Networks (LSTM RNNs) to recognize fonts from images. This hybrid architecture is chosen for its ability to capture both spatial features from images (via CNNs) and temporal dependencies within sequences of features (via LSTM RNNs), making it well-suited for the complex task of font recognition.

The project workflow includes the following stages:

Data Acquisition and Preprocessing: A comprehensive dataset comprising images containing text written in various fonts is collected. These images are preprocessed to standardize their size, orientation, and format. Techniques such as resizing, normalization, and augmentation may be applied to enhance the diversity and quality of the training data.

Feature Extraction with CNNs: Convolutional Neural Networks are utilized to extract hierarchical spatial features from the preprocessed images. The CNN architecture consists of multiple convolutional layers followed by pooling layers, enabling the model to learn abstract representations of font styles from the input images.

Sequence Modeling with LSTM RNNs: The extracted CNN features are fed into a Long Short-Term Memory Recurrent Neural Network, which is designed to capture temporal dependencies in sequences of feature vectors. The LSTM RNN processes the feature sequences to learn the contextual information and temporal patterns associated with different font styles.

Model Training: The combined CNN-LSTM RNN architecture is trained on the labeled dataset using supervised learning techniques. The model learns to map input images to corresponding font labels by minimizing a suitable loss function, such as categorical cross-entropy.

Evaluation and Optimization: The trained model is evaluated using a separate validation dataset to assess its performance in font recognition tasks. Metrics such as accuracy, precision, recall, and F1-score are computed to measure the model's effectiveness. Hyperparameter tuning and optimization techniques may be applied to improve the model's performance further.

The Font Recognition project leveraging CNN-LSTM RNN architecture offers several advantages, including:

Ability to capture both spatial and temporal features for comprehensive font recognition.

Enhanced performance in handling complex font styles and variations.

Flexibility for integration into various applications and platforms.

Potential for scalability and adaptation to new font styles through continued training and refinement.

Overall, by combining CNNs and LSTM RNNs, the Font Recognition project aims to provide a robust and accurate solution for automatic font identification in diverse image datasets.
