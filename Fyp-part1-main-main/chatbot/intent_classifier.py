import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import json
import os

class IntentClassifier:
    """Machine learning based intent classifier for chatbot"""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=500, stop_words='english', ngram_range=(1, 2))
        self.classifier = None
        self.classes = None
        self.model_path = 'models/saved_models/intent_classifier.pkl'
        
    def create_training_data(self):
        """Create training data for intent classification"""
        training_data = []
        
        # Greeting intents
        greeting_examples = [
            "hello", "hi", "hey", "good morning", "good afternoon",
            "good evening", "how are you", "what's up"
        ]
        for text in greeting_examples:
            training_data.append({'text': text, 'intent': 'greeting'})
        
        # Sepsis risk intents
        risk_examples = [
            "what is my sepsis risk", "am i at risk of sepsis",
            "do i have sepsis", "check my sepsis risk",
            "predict sepsis", "sepsis probability",
            "what are my chances of sepsis", "risk assessment"
        ]
        for text in risk_examples:
            training_data.append({'text': text, 'intent': 'sepsis_risk'})
        
        # Explanation intents
        explanation_examples = [
            "why am i at risk", "explain the prediction",
            "what factors affect my risk", "how was this predicted",
            "why this result", "what does this mean",
            "explain sepsis risk", "reason for prediction"
        ]
        for text in explanation_examples:
            training_data.append({'text': text, 'intent': 'explanation'})
        
        # Symptoms intents
        symptom_examples = [
            "what are sepsis symptoms", "symptoms of sepsis",
            "how do i know if i have sepsis", "signs of sepsis",
            "what to look for", "early symptoms",
            "warning signs", "how to recognize sepsis"
        ]
        for text in symptom_examples:
            training_data.append({'text': text, 'intent': 'symptoms'})
        
        # Prevention intents
        prevention_examples = [
            "how to prevent sepsis", "prevention tips",
            "avoid sepsis", "reduce risk",
            "what can i do to prevent", "preventive measures",
            "stop sepsis", "lower my risk"
        ]
        for text in prevention_examples:
            training_data.append({'text': text, 'intent': 'prevention'})
        
        # Treatment intents
        treatment_examples = [
            "how is sepsis treated", "treatment for sepsis",
            "what is the cure", "medication for sepsis",
            "hospital treatment", "what do doctors do",
            "therapy for sepsis", "management of sepsis"
        ]
        for text in treatment_examples:
            training_data.append({'text': text, 'intent': 'treatment'})
        
        # Help intents
        help_examples = [
            "help", "what can you do", "how can you help me",
            "assist me", "support", "guide me",
            "what are your capabilities", "how do i use this"
        ]
        for text in help_examples:
            training_data.append({'text': text, 'intent': 'help'})
        
        # Goodbye intents
        goodbye_examples = [
            "bye", "goodbye", "see you", "thank you",
            "thanks", "farewell", "take care", "that's all"
        ]
        for text in goodbye_examples:
            training_data.append({'text': text, 'intent': 'goodbye'})
        
        # Create variations
        variations = self._create_variations(training_data)
        training_data.extend(variations)
        
        return training_data
    
    def _create_variations(self, training_data):
        """Create variations of training examples"""
        variations = []
        
        for item in training_data:
            text = item['text']
            intent = item['intent']
            
            # Add question variations
            if not text.endswith('?'):
                variations.append({'text': text + '?', 'intent': intent})
            
            # Add capitalization variations
            variations.append({'text': text.capitalize(), 'intent': intent})
            variations.append({'text': text.upper(), 'intent': intent})
            
            # Add "please" variations
            if not text.startswith('please'):
                variations.append({'text': 'please ' + text, 'intent': intent})
        
        return variations
    
    def preprocess_text(self, text):
        """Preprocess text for classification"""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def train(self, save_model=True):
        """Train the intent classifier"""
        print("Training intent classifier...")
        
        # Create training data
        training_data = self.create_training_data()
        
        # Convert to DataFrame
        df = pd.DataFrame(training_data)
        
        # Preprocess text
        df['processed_text'] = df['text'].apply(self.preprocess_text)
        
        # Split data
        X = df['processed_text']
        y = df['intent']
        
        # Vectorize text
        X_vectorized = self.vectorizer.fit_transform(X)
        
        # Get classes
        self.classes = y.unique()
        
        # Train classifier
        self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.classifier.fit(X_vectorized, y)
        
        # Evaluate
        X_train, X_test, y_train, y_test = train_test_split(
            X_vectorized, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train final model on all data
        self.classifier.fit(X_vectorized, y)
        
        # Test performance
        y_pred = self.classifier.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        print("Intent classifier training completed.")
        print(f"Accuracy: {report['accuracy']:.3f}")
        
        # Save model
        if save_model:
            self.save_model()
        
        return report
    
    # ADD THIS METHOD - This is what the chatbot module expects
    def predict_intent(self, text, threshold=0.5):
        """Predict intent for given text (compatibility method)"""
        result = self.predict(text, threshold)
        return {
            'intent': result['intent'],
            'confidence': result['confidence'],
            'processed_text': self.preprocess_text(text)  # Add processed text
        }
    
    # ADD THIS METHOD - This is what the chatbot module expects
    def get_response_suggestions(self, intent):
        """Get response suggestions for intent (compatibility method)"""
        suggestions = {
            'greeting': [
                "Hello! I'm your sepsis prediction assistant. How can I help you today?",
                "Hi there! I can help you with sepsis risk assessment and information.",
                "Welcome! Ask me about sepsis symptoms, prevention, or risk assessment."
            ],
            'sepsis_risk': [
                "I can help assess sepsis risk. Would you like to provide some symptoms or vital signs?",
                "For sepsis risk assessment, I need information like temperature, heart rate, and lab results.",
                "You can upload patient data or enter symptoms for a sepsis risk prediction."
            ],
            'explanation': [
                "I can explain sepsis predictions by showing which features contributed most to the risk assessment.",
                "The prediction explanation shows vital signs and lab values that increased or decreased sepsis risk.",
                "I'll provide a detailed breakdown of factors influencing the sepsis prediction."
            ],
            'symptoms': [
                "Common sepsis symptoms include: fever, rapid heart rate, rapid breathing, confusion, and extreme pain.",
                "Look for: temperature >38°C or <36°C, heart rate >90 bpm, respiratory rate >20 bpm.",
                "Sepsis symptoms often include chills, low blood pressure, and abnormal white blood cell count."
            ],
            'prevention': [
                "Prevent sepsis by: treating infections promptly, practicing good hygiene, getting vaccinated.",
                "Early treatment of infections and proper wound care can prevent sepsis.",
                "Hospital prevention: hand hygiene, proper catheter care, and antibiotic stewardship."
            ],
            'treatment': [
                "Sepsis treatment requires immediate medical attention: antibiotics, IV fluids, and hospital care.",
                "Treatment includes: antibiotics, source control, vasopressors, and supportive care.",
                "Early recognition and treatment in a hospital setting is crucial for sepsis survival."
            ],
            'help': [
                "I can help you with: 1) Sepsis risk assessment 2) Symptom information 3) Prevention tips 4) Prediction explanations",
                "You can: upload patient data, enter symptoms manually, or ask questions about sepsis.",
                "Try asking about symptoms, requesting a risk assessment, or getting prevention advice."
            ],
            'goodbye': [
                "Goodbye! Stay healthy and remember to seek medical help for any sepsis concerns.",
                "Take care! Don't hesitate to return if you have more questions about sepsis.",
                "Thank you for using the sepsis assistant. Stay safe!"
            ],
            'general_query': [
                "I'm here to help with sepsis-related questions. What would you like to know?",
                "I can assist with sepsis risk assessment, symptoms, prevention, and treatment information.",
                "Please ask me about sepsis or request a risk assessment."
            ]
        }
        
        return suggestions.get(intent, ["I'm here to help with sepsis-related questions. What would you like to know?"])
    
    def predict(self, text, threshold=0.5):
        """Predict intent for given text"""
        if self.classifier is None:
            self.load_model()
        
        # Preprocess text
        processed_text = self.preprocess_text(text)
        
        # Vectorize
        X = self.vectorizer.transform([processed_text])
        
        # Predict
        probabilities = self.classifier.predict_proba(X)[0]
        predicted_class = self.classifier.predict(X)[0]
        confidence = max(probabilities)
        
        # Get top 3 predictions
        class_indices = np.argsort(probabilities)[::-1]
        top_predictions = []
        
        for idx in class_indices[:3]:
            intent = self.classes[idx]
            prob = probabilities[idx]
            top_predictions.append({
                'intent': intent,
                'confidence': float(prob)
            })
        
        # Return prediction if confidence meets threshold
        if confidence >= threshold:
            return {
                'intent': predicted_class,
                'confidence': float(confidence),
                'all_predictions': top_predictions
            }
        else:
            return {
                'intent': 'general_query',
                'confidence': float(confidence),
                'all_predictions': top_predictions
            }
    
    def save_model(self):
        """Save trained model"""
        if self.classifier is None:
            raise ValueError("Model must be trained before saving")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        
        # Save model and vectorizer
        model_data = {
            'classifier': self.classifier,
            'vectorizer': self.vectorizer,
            'classes': self.classes
        }
        
        joblib.dump(model_data, self.model_path)
        print(f"Intent classifier saved to {self.model_path}")
    
    def load_model(self):
        """Load trained model"""
        try:
            if os.path.exists(self.model_path):
                model_data = joblib.load(self.model_path)
                self.classifier = model_data['classifier']
                self.vectorizer = model_data['vectorizer']
                self.classes = model_data['classes']
                print(f"Intent classifier loaded from {self.model_path}")
                return True
            else:
                # Train a new model if none exists
                print("No saved intent classifier found. Training new one...")
                self.train()
                return True
        except Exception as e:
            print(f"Could not load intent classifier: {e}")
            # Train a new model if loading fails
            print("Training new intent classifier...")
            self.train()
            return True
    
    def get_intent_description(self, intent):
        """Get description for intent"""
        descriptions = {
            'greeting': 'Greeting or opening conversation',
            'sepsis_risk': 'Query about sepsis risk assessment',
            'explanation': 'Request for explanation of prediction',
            'symptoms': 'Question about sepsis symptoms',
            'prevention': 'Information about sepsis prevention',
            'treatment': 'Information about sepsis treatment',
            'help': 'Request for help or information about capabilities',
            'goodbye': 'Ending conversation',
            'general_query': 'General or unclear query'
        }
        
        return descriptions.get(intent, 'Unknown intent')
    
    def analyze_query_patterns(self, queries):
        """Analyze patterns in user queries for improvement"""
        if not queries:
            return {}
        
        # Count intents
        intent_counts = {}
        for query in queries:
            result = self.predict(query['text'])
            intent = result['intent']
            intent_counts[intent] = intent_counts.get(intent, 0) + 1
        
        # Calculate percentages
        total = len(queries)
        intent_percentages = {}
        for intent, count in intent_counts.items():
            intent_percentages[intent] = (count / total) * 100
        
        return {
            'total_queries': total,
            'intent_distribution': intent_counts,
            'intent_percentages': intent_percentages,
            'most_common_intent': max(intent_counts, key=intent_counts.get) if intent_counts else None
        }
    
    def add_training_example(self, text, intent, retrain=False):
        """Add new training example"""
        # Load existing training data
        try:
            with open('models/saved_models/training_data.json', 'r') as f:
                training_data = json.load(f)
        except:
            training_data = []
        
        # Add new example
        training_data.append({
            'text': text,
            'intent': intent,
            'timestamp': pd.Timestamp.now().isoformat()
        })
        
        # Save updated training data
        with open('models/saved_models/training_data.json', 'w') as f:
            json.dump(training_data, f, indent=2)
        
        # Retrain if requested
        if retrain:
            self.train(save_model=True)
        
        return True