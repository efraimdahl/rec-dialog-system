# rec-dialog-system
Task 1b of the restaurant dialog system.


## Installation instructions
1. In your IDE's console (from the project's directory), run ".bin/activate" to activate the virtual environment.
2. Run "pip install -r requirements.txt" to install all dependencies.
3. Run state-machines.py to enter the user-chatbot loop and find a restaurant!


## Classifier
1. The Text-Parser is using a decision tree classifier from assignment 1a.
2. Set your prefered classifier by editing the classifier variable in the main function of the state-machine.py file.
3. If you want to train the classifier and create the pkl files that contain the model from scratch, do the following:
    1. cd ass_1a //This is important because otherwise the local filepath definitions in the python files won't work.
    1. run $python main.py
