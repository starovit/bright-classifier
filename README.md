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
