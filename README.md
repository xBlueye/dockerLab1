# dockerLab1

Container 1 trains models and prints out the accuracy on the training set then saves the models as multiple pickle files

Container 2 loads the pickle files and prints predictions based on the test sets.

unfortunatly I couldn't turn this into an interactable app because the input data consists of timeseries and is thus rather complex.

To use :
1) create a volume (to be attached at /app/shared)
2) run the first image
3) run the second image 
