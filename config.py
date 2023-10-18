#Configuration setting for User test, system, use A or B, system A has no utterances, system B has thinking utterances
CHOSEN_SYSTEM = "B"

# Task: Ask user about correctness of match for Levenshtein results
ASK_CONFIRMATION_LEVENSHTEIN = False

# Task: Allow multiple preferences per utterance
ALLOW_MULTIPLE_PREFERENCES_PER_UTTERANCE = True

# Task: Allow preferences to be stated in random order or not. 
# If False they can be stated in a single or multiple utterances, but have to appear in order. The order is are, foodType, priceRange
RANDOMIZE_PREFERENCE_QUESTION_ORDER = True

# Task: Delay before responding (in seconds)
RESPONSE_DELAY = 0

# Task: Output in all caps
ALL_CAPS_RESPONSE = False

# Task: Use text-to-speech for system utterances
TTS = True

#Not a task, but triggers a small test-suite to run on program start.
DIALOG_TESTING = False
# Random other chatbot settings
TYPING_SPEED_DELAY = 0.2#0.03
COLORED_OUTPUT = True