# rec-dialog-system
Task 1c of the restaurant dialog system.


## Installation instructions
1. In your IDE's console (from the project's directory), run ".bin/activate" to activate the virtual environment.
2. Run "pip install -r requirements.txt" to install all dependencies.
3. Run state-machines.py to enter the user-chatbot loop and find a restaurant!


## Classifier
1. The Text-Parser is using a decision tree classifier from assignment 1a.
2. Set your prefered classifier by editing the classifier variable in the main function of the state-machine.py file.
3. If you want to train the classifier and create the models from scratch, do the following:
    1. cd ass_1a //This is important because otherwise the local filepath definitions in the python files won't work.
    1. run $python main.py

## File structure and description
- ass_1a/ - Contains files used to evaluate different classifiers during assignment 1a.
- - results/ - Contains the results of the models' evaluation.
- - evaluation.py - Contains functions that test the performance of different classfifiers.
- - keyword_model.py - Contains the KeywordClassifier class
- - **main.py** - Run this file to train and evaluate each model, and dump the results in ass_1a/results/.
- - preprocessing.py - Contains code to preprocess the dataset.
- - training.py - Contains functions to train each model.
- data/ - Contains all files relating to data.
- deprecated/ - Contains files that are not used anymore, but are still relevant to the project report.
- config.py - Contains constants relating to the chatbot's configuration (as part of assignment 1c's configurability tasks).
- **state_machine.py** - The main file of the chatbot. Run this file to enter the user input loop and activate the chatbot. Contains all functions relating to the state machine.
- text_parser.py - Contains functions that are used to parse the text.
- utils.py - Contains helper functions that fulfill small tasks which do not make sense to put in a specific file.
