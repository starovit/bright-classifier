from terminal_classifier import read_folder, bath_predict

images, _ = read_folder("example_data/")
predictions = bath_predict(images,
                           model_type="nn",
                           return_type="proba") # list of np.arrays as input
print(predictions)