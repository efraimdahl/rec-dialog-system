class KeywordClassifier:
    """Classifies sentences based on keyword matching."""
    
    def __init__(self) -> None:
        """Initialize a new instance of the KeywordClassifier class.
        """
        self.keywords = {
            'ack': ['okay', 'um'],
            'affirm': ['yes', 'right'],
            'bye': ['see you', 'good bye'],
            'confirm': ['is it', 'does it', 'are you'],
            'deny': ['dont want', 'no thanks', 'not interested', 'no'],
            'hello': ['hi', 'hello', 'hey', 'whazzup'],
            'inform': ['im looking for', 'i want', 'i need', 'i could use'],
            'negate': ['no', 'not', 'none'],
            'null': ['cough', 'umm', 'hm' 'hmm', 'uh', 'uhh', 'um'],
            'repeat': ['repeat', 'say again', 'say that again'],
            'reqalts': ['how about', 'other option', 'alternative'],
            'reqmore': ['more', 'else', 'additional'],
            'request': ['what is', 'can you tell me', 'please provide'],
            'restart': ['start over', 'reset', 'begin again'],
            'thankyou': ['thank you', 'thanks', 'appreciate', 'cheers']
        }
        
    def __repr__(self) -> str:
        return "KeywordClassifier()"

    def classify(self, sentence: str) -> str:
        """Classifies a single instance.

        Args:
            sentence (str): The input sentence to be classified.

        Returns:
            str: A label corresponding to the predicted dialog act.
        """
        # Lowercase the sentence
        sentence = sentence.lower()

        # Loop through all dialog acts and return the first match
        for act, act_keywords in self.keywords.items():
            for keyword in act_keywords:
                if keyword in sentence:
                    return act
        return 'null'

    def predict(self, X: list) -> list:
        """Predicts the dialog acts for a list of sentences.

        Args:
            X (list): Listlike object containing sentences to be classified.

        Returns:
            list: List of labels corresponding to the predicted dialog acts.
        """
        return [self.classify(sentence) for sentence in X]
