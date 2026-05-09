import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib
import numpy as np

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

class NLPProcessor:
    """Processes natural language queries for the chatbot"""
    
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.intent_classifier = None
        self.vectorizer = TfidfVectorizer(max_features=1000)
        self.intent_patterns = self._load_intent_patterns()
        
        # Try to load spaCy model, fallback to simple processing
        try:
            self.nlp = spacy.load("en_core_web_sm")
            self.use_spacy = True
        except:
            self.use_spacy = False
            print("spaCy model not found. Using simple NLP processing.")
    
    def _load_intent_patterns(self):
        """Load patterns for intent recognition"""
        return {
            'greeting': [
                r'hello', r'hi', r'hey', r'good morning', r'good afternoon',
                r'good evening', r'how are you'
            ],
            'sepsis_risk': [
                r'risk', r'predict', r'sepsis', r'chance', r'probability',
                r'what is my risk', r'am i at risk', r'do i have sepsis'
            ],
            'explanation': [
                r'why', r'explain', r'reason', r'cause', r'factor',
                r'how come', r'what caused', r'why am i'
            ],
            'symptoms': [
                r'symptom', r'sign', r'indication', r'feel', r'experience',
                r'what are the symptoms', r'how do i know'
            ],
            'prevention': [
                r'prevent', r'avoid', r'stop', r'reduce risk',
                r'how to prevent', r'what can i do'
            ],
            'treatment': [
                r'treatment', r'cure', r'medicine', r'drug', r'therapy',
                r'how is it treated', r'what is the treatment'
            ],
            'help': [
                r'help', r'assist', r'support', r'what can you do',
                r'how can you help'
            ],
            'goodbye': [
                r'bye', r'goodbye', r'see you', r'thank you', r'thanks'
            ]
        }
    
    def preprocess_text(self, text):
        """Preprocess input text"""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and extra spaces
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        if self.use_spacy:
            # Use spaCy for advanced processing
            doc = self.nlp(text)
            tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
        else:
            # Use NLTK for basic processing
            tokens = word_tokenize(text)
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                     if token not in self.stop_words and token.isalpha()]
        
        return ' '.join(tokens)
    
    def extract_intent(self, text):
        """Extract intent from user query"""
        cleaned_text = self.preprocess_text(text)
        
        # Check patterns for each intent
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, cleaned_text, re.IGNORECASE):
                    return intent
        
        # If no pattern matches, use keyword matching
        keywords = cleaned_text.split()
        
        sepsis_keywords = ['sepsis', 'infection', 'blood', 'poisoning']
        if any(keyword in sepsis_keywords for keyword in keywords):
            return 'sepsis_risk'
        
        question_words = ['what', 'how', 'why', 'when', 'where']
        if any(word in keywords for word in question_words):
            return 'explanation'
        
        return 'general_query'
    
    def extract_entities(self, text):
        """Extract entities from text"""
        entities = {
            'symptoms': [],
            'measurements': [],
            'time_periods': []
        }
        
        # Simple entity extraction
        symptom_patterns = [
            r'fever', r'chills', r'shivering', r'pain', r'ache',
            r'confus', r'dizzy', r'breath', r'rapid heartbeat',
            r'sweat', r'cold', r'clammy'
        ]
        
        measurement_patterns = [
            r'\d+\s*(degrees|°C|°F)',  # Temperature
            r'\d+\s*(bpm|beats)',      # Heart rate
            r'\d+\s*(mmhg|pressure)',  # Blood pressure
            r'\d+\s*(breaths|respirations)'  # Respiratory rate
        ]
        
        time_patterns = [
            r'\d+\s*(hours?|hrs?)',
            r'\d+\s*(days?|d)',
            r'\d+\s*(weeks?|wks?)'
        ]
        
        # Find matches
        for pattern in symptom_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                entities['symptoms'].append(pattern.strip(r'\\'))
        
        for pattern in measurement_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities['measurements'].extend(matches)
        
        for pattern in time_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities['time_periods'].extend(matches)
        
        return entities
    
    def train_intent_classifier(self, training_data):
        """Train intent classifier if needed"""
        texts = [item['text'] for item in training_data]
        intents = [item['intent'] for item in training_data]
        
        # Preprocess texts
        processed_texts = [self.preprocess_text(text) for text in texts]
        
        # Vectorize
        X = self.vectorizer.fit_transform(processed_texts)
        
        # Train classifier
        self.intent_classifier = MultinomialNB()
        self.intent_classifier.fit(X, intents)
        
        # Save classifier
        joblib.dump({
            'classifier': self.intent_classifier,
            'vectorizer': self.vectorizer
        }, 'models/saved_models/intent_classifier.pkl')
        
        return self.intent_classifier
    
    def predict_intent(self, text):
        """Predict intent using trained classifier"""
        if self.intent_classifier is None:
            # Fallback to pattern matching
            return self.extract_intent(text)
        
        # Preprocess and vectorize
        processed_text = self.preprocess_text(text)
        X = self.vectorizer.transform([processed_text])
        
        # Predict
        intent = self.intent_classifier.predict(X)[0]
        probability = self.intent_classifier.predict_proba(X).max()
        
        return intent if probability > 0.5 else 'general_query'
    
    def load_intent_classifier(self, path='models/saved_models/intent_classifier.pkl'):
        """Load trained intent classifier"""
        try:
            loaded = joblib.load(path)
            self.intent_classifier = loaded['classifier']
            self.vectorizer = loaded['vectorizer']
            return True
        except:
            print("Could not load intent classifier. Using pattern matching.")
            return False