# Transfer-Learning-and-Image-Classification-with-CNNs

Experiment 1: Train a CNN model to distinguish between cats and dogs. Use a learning rate of 0.0001 instead of the one that is used in the tutorial
Train a model (with the same architecture as the model in the tutorial) on the Stanford Dogs dataset: https://www.tensorflow.org/datasets/catalog/stanford_dogsLinks to an external site.. When trained, save this model to a file.
Experiment 2: Load the saved model and replace only the output layer of the model (to align it to the new problem). When this is done, train and evaluate the model (for 50 epochs) on the cats and dogs dataset.
Experiment 3: Load the saved model and replace the output layer of the model, as well as the first two convolutional layers (keep the weights of all other layers). Train and evaluate the model on the cats and dogs dataset when this is done.
Experiment 4: Load the saved model and replace the output layer of the model, as well as the two last convolutional layers. Train and evaluate the model on the cats and dogs dataset when this is done.
