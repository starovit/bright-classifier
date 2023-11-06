# Bright Classifier
Utilizing MediaPipe with Logistic Regression for assessing image brightness quality classification.

## Introduction
The `bright-classifier` is a tool designed to classify images based on the brightness quality, by using MediaPipe for image processing and Logistic Regression for predictions.

## Usage Cases
### 1. Terminal Execution
Run the classifier directly from the terminal by specifying the path to the images.
```bash
python terminal_classifier.py example_data/
```
#### Description
- `example_data/`: path to the directory that with images for classification
#### Output
- `predictions.json`: JSON file will contain the classification probabilities for each image processed.


### 2. Integration with External Code
You can incorporate the `batch_predict` function from `terminal_classifier.py` into your own codebase.
```python
from terminal_classifier import batch_predict
# Assuming `images` is a list of RGB images represented as NumPy arrays
predictions = batch_predict(images)
```
#### Input:
A list of RGB images, each represented as a NumPy array.
#### Output:
A list of predictions, with each corresponding to the class_1 probability of the image being of good brightness quality.
#### Usage Example
For a implementation sample, refer to _example.py_ in the repository.


## Files in the Repository:
1. `./example_data` - Contains `.jpg` images for testing, labeled as class_0 or class_1.
2. `prediction.json` - An example of prediction output.
3. `./models` - Directory with the MediaPipe and Logistic Regression model (`logreg`) saved as a pickle file.
4. `mputils.py` - Contains utilities for working with MediaPipe.
5. `terminal_classifier.py` - The script for running classification from the terminal.
6. `example.py` - Provides an example of using `batch_prediction` function.





