import random
from datetime import datetime
from utils.constants import CHATBOT_INTENTS, RISK_LEVELS
import json

class ResponseGenerator:
    """Generates responses for the chatbot"""
    
    def __init__(self):
        self.responses = self._load_responses()
        self.context = {}
    
    def _load_responses(self):
        """Load response templates"""
        return {
            CHATBOT_INTENTS['GREETING']: [
                "Hello! I'm your sepsis prediction assistant. How can I help you today?",
                "Hi there! I'm here to help with sepsis prediction and information. What would you like to know?",
                "Welcome! I can help you understand sepsis risk and prevention. How can I assist you?"
            ],
            CHATBOT_INTENTS['SEPSIS_RISK']: [
                "Based on your vital signs and lab results, your sepsis risk is {risk_level} ({probability}%). {explanation}",
                "The analysis shows a {risk_level} risk of sepsis ({probability}%). Key factors include: {factors}",
                "Your sepsis risk assessment indicates {risk_level} probability ({probability}%). {recommendation}"
            ],
            CHATBOT_INTENTS['EXPLANATION']: [
                "The prediction is based on several factors: {factors}. The most important ones are {top_factors}.",
                "Here's why you're at {risk_level} risk: {explanation}. The main contributors are {contributors}.",
                "The model considers {factor_count} clinical parameters. Key indicators for your case: {indicators}"
            ],
            CHATBOT_INTENTS['SYMPTOMS']: [
                "Common sepsis symptoms include: fever, chills, rapid breathing, rapid heart rate, confusion, and extreme pain. If you experience these, seek immediate medical attention.",
                "Watch for: high temperature or low body temperature, fast heartbeat, rapid breathing, severe shivering, and confusion. These could indicate sepsis.",
                "Sepsis symptoms often include: fever above 101°F or below 96.8°F, heart rate above 90 beats per minute, breathing rate above 20 breaths per minute, and altered mental state."
            ],
            CHATBOT_INTENTS['PREVENTION']: [
                "To prevent sepsis: practice good hygiene, get vaccinated, properly clean wounds, manage chronic conditions, and seek prompt treatment for infections.",
                "Prevention tips: wash hands regularly, keep wounds clean, follow vaccination schedules, manage diabetes carefully, and don't ignore infections.",
                "Reduce sepsis risk by: maintaining good hygiene, promptly treating infections, properly caring for wounds, and managing underlying health conditions."
            ],
            CHATBOT_INTENTS['TREATMENT']: [
                "Sepsis treatment typically involves: antibiotics, intravenous fluids, oxygen therapy, and sometimes surgery to remove infection sources. Early treatment is crucial.",
                "Treatment includes: broad-spectrum antibiotics, fluid resuscitation, vasopressors if needed, and source control. Hospital care is essential.",
                "For sepsis: immediate antibiotics, intravenous fluids, monitoring vital signs, and treating the underlying infection. ICU care may be necessary."
            ],
            CHATBOT_INTENTS['HELP']: [
                "I can help you with: sepsis risk prediction, understanding prediction explanations, learning about symptoms, prevention tips, and treatment information.",
                "I can: assess sepsis risk, explain risk factors, provide information about sepsis symptoms and prevention, and answer related questions.",
                "My capabilities include: predicting sepsis risk, explaining the predictions, providing sepsis information, and answering your questions about sepsis."
            ],
            CHATBOT_INTENTS['GOODBYE']: [
                "Goodbye! Stay healthy and remember to seek medical attention if you experience concerning symptoms.",
                "Take care! Don't hesitate to consult a healthcare professional if you have concerns.",
                "Farewell! Remember that early detection and treatment are key for sepsis."
            ],
            'general_query': [
                "I understand you're asking about '{query}'. Could you rephrase or ask about sepsis prediction, symptoms, or prevention?",
                "I'm specialized in sepsis-related information. Could you ask about sepsis risk, symptoms, or treatment?",
                "For sepsis-related questions, I can help with predictions, explanations, symptoms, and prevention."
            ],
            'error': [
                "I'm having trouble processing your request. Could you please rephrase your question?",
                "I didn't understand that. Could you ask about sepsis prediction or information?",
                "My apologies, I couldn't process that. Please try asking about sepsis risk or related topics."
            ]
        }
    
    def generate_response(self, intent, entities=None, context=None, original_message=None):
        """Generate response based on intent and context"""
        if intent not in self.responses:
            intent = 'general_query'
        
        # Get base response template
        template = random.choice(self.responses[intent])
        
        # Prepare context data
        response_data = {
            'intent': intent,
            'timestamp': datetime.now().isoformat(),
            'entities': entities or {},
            'context': context or {}
        }
        
        # Fill template with context data
        filled_response = self._fill_template(template, context)
        
        # Generate full response object
        response = {
            'response': filled_response,
            'data': response_data,
            'suggestions': self._generate_suggestions(intent)
        }
        
        return response
    
    def _fill_template(self, template, context):
        """Fill response template with context data"""
        if not context:
            return template
        
        # Replace placeholders with context data
        if '{risk_level}' in template and 'risk_level' in context:
            template = template.replace('{risk_level}', context['risk_level'])
        
        if '{probability}' in template and 'probability' in context:
            probability = context.get('probability', 0) * 100
            template = template.replace('{probability}', f"{probability:.1f}")
        
        if '{factors}' in template and 'factors' in context:
            factors = context.get('factors', [])
            if isinstance(factors, list) and factors:
                factors_str = ', '.join(factors[:3])
                template = template.replace('{factors}', factors_str)
        
        if '{explanation}' in template and 'explanation' in context:
            explanation = context.get('explanation', 'Please consult the detailed explanation in the dashboard.')
            template = template.replace('{explanation}', explanation)
        
        if '{recommendation}' in template and 'recommendations' in context:
            recommendations = context.get('recommendations', [])
            if recommendations:
                template = template.replace('{recommendation}', recommendations[0])
        
        if '{top_factors}' in template and 'top_factors' in context:
            top_factors = context.get('top_factors', [])
            if top_factors:
                factors_str = ', '.join([f['feature'] for f in top_factors[:2]])
                template = template.replace('{top_factors}', factors_str)
        
        if '{contributors}' in template and 'contributors' in context:
            contributors = context.get('contributors', [])
            if contributors:
                template = template.replace('{contributors}', ', '.join(contributors[:3]))
        
        if '{indicators}' in template and 'indicators' in context:
            indicators = context.get('indicators', [])
            if indicators:
                template = template.replace('{indicators}', ', '.join(indicators[:3]))
        
        if '{factor_count}' in template and 'factor_count' in context:
            template = template.replace('{factor_count}', str(context['factor_count']))
        
        if '{query}' in template:
            template = template.replace('{query}', 'your question')
        
        return template
    
    def _generate_suggestions(self, intent):
        """Generate follow-up suggestions based on intent"""
        suggestions = {
            CHATBOT_INTENTS['GREETING']: [
                "What is my sepsis risk?",
                "What are sepsis symptoms?",
                "How can I prevent sepsis?"
            ],
            CHATBOT_INTENTS['SEPSIS_RISK']: [
                "Why am I at this risk level?",
                "What factors affect my risk?",
                "What should I do next?"
            ],
            CHATBOT_INTENTS['EXPLANATION']: [
                "What are the key risk factors?",
                "How can I lower my risk?",
                "What symptoms should I watch for?"
            ],
            CHATBOT_INTENTS['SYMPTOMS']: [
                "What is the treatment for sepsis?",
                "How is sepsis diagnosed?",
                "When should I seek help?"
            ],
            CHATBOT_INTENTS['PREVENTION']: [
                "What are the early signs?",
                "Who is at high risk?",
                "How is sepsis treated?"
            ],
            CHATBOT_INTENTS['TREATMENT']: [
                "What are the recovery expectations?",
                "How can recurrence be prevented?",
                "What are the complications?"
            ],
            CHATBOT_INTENTS['HELP']: [
                "Predict my sepsis risk",
                "Explain sepsis symptoms",
                "Tell me about prevention"
            ]
        }
        
        return suggestions.get(intent, [
            "What is sepsis?",
            "How is sepsis predicted?",
            "What are risk factors?"
        ])
    
    def generate_risk_response(self, prediction_result, explanation=None):
        """Generate response for risk prediction"""
        risk_level = prediction_result.get('risk_level', 'Unknown')
        probability = prediction_result.get('probability', 0)
        
        # Get risk description
        risk_descriptions = {
            'Low Risk': "This suggests a low likelihood of sepsis. Continue monitoring standard vital signs.",
            'Medium Risk': "This indicates moderate risk. Closer monitoring and assessment are recommended.",
            'High Risk': "This suggests high likelihood of sepsis. Immediate clinical assessment is advised."
        }
        
        # Get factors from explanation
        factors = []
        if explanation and 'feature_effects' in explanation:
            factors = [effect['feature'].replace('_', ' ').title() 
                      for effect in explanation['feature_effects'][:3]]
        
        # Prepare context
        context = {
            'risk_level': risk_level,
            'probability': probability,
            'explanation': risk_descriptions.get(risk_level, ''),
            'factors': factors,
            'recommendations': self._get_recommendations(risk_level)
        }
        
        return self.generate_response(CHATBOT_INTENTS['SEPSIS_RISK'], context=context)
    
    def _get_recommendations(self, risk_level):
        """Get recommendations based on risk level"""
        recommendations = {
            'Low Risk': [
                "Continue regular monitoring",
                "Maintain good hygiene practices",
                "Report any new symptoms promptly"
            ],
            'Medium Risk': [
                "Increase monitoring frequency",
                "Consider additional lab tests",
                "Consult healthcare provider"
            ],
            'High Risk': [
                "Seek immediate medical attention",
                "Initiate sepsis protocol if available",
                "Prepare for possible hospital admission"
            ]
        }
        
        return recommendations.get(risk_level, [
            "Consult with healthcare provider",
            "Monitor symptoms closely",
            "Follow medical advice"
        ])
    
    def update_context(self, key, value):
        """Update chatbot context"""
        self.context[key] = value
    
    def clear_context(self):
        """Clear chatbot context"""
        self.context = {}
    
    def generate_fallback_response(self, original_message):
        """Generate fallback response when intent is not recognized"""
        # Try to extract keywords
        keywords = self._extract_keywords(original_message)
        
        if keywords:
            response = f"I see you mentioned {', '.join(keywords[:3])}. "
            response += "I can help with sepsis-related information. Could you rephrase your question?"
        else:
            response = random.choice(self.responses['general_query'])
        
        return {
            'response': response,
            'data': {
                'intent': 'fallback',
                'original_message': original_message,
                'keywords': keywords
            },
            'suggestions': [
                "What is sepsis?",
                "How is sepsis predicted?",
                "What are the symptoms?"
            ]
        }
    
    def _extract_keywords(self, text):
        """Extract keywords from text"""
        if not text:
            return []
        
        # Common medical keywords
        medical_keywords = [
            'fever', 'infection', 'blood', 'pressure', 'heart', 'rate',
            'temperature', 'breathing', 'pain', 'hospital', 'doctor',
            'medicine', 'antibiotic', 'test', 'lab', 'result'
        ]
        
        text_lower = text.lower()
        found_keywords = [kw for kw in medical_keywords if kw in text_lower]
        
        return found_keywords