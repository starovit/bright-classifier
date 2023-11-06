# bright-classifier
Media-pipe + LogisticRegression image brightness quality classification

## Usage Cases
**1. Terminal run example:** <br/>
_python terminal_classifier.py example_data/_ <br/><br/>
Description: example_data/ - path to images <br/>
Output: prediction.json file <br/>

**2. Use predict function in external code:** <br/>
_batch_predict_(images) function from _terminal_classifier.py_ <br/>
Input: list of RGB image (np.array) <br/>
Output: list of class_1 predictions <br/>
Usage example: example.py



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

**Description**
- `example_data/`: This is the path to the directory that contains the images which will be classified.
**Output**
- `predictions.json`: This JSON file will contain the classification probabilities for each image processed.

## 2. Integration with External Code
You can incorporate the `batch_predict` function from `terminal_classifier.py` into your own codebase.

```python
from terminal_classifier import batch_predict
# Assuming `images` is a list of RGB images represented as NumPy arrays
predictions = batch_predict(images)
```

Input:
A list of RGB images, each represented as a NumPy array.
Output:
A list of predictions, with each corresponding to the class_1 probability of the image being of good brightness quality.
Usage Example:
Refer to the example.py for a detailed implementation sample.



