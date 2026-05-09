from chatbot.document_loader import load_documents
from chatbot.simple_retriever import find_relevant_passage


class SepsisDocAgent:
    def __init__(self):
        self.documents = load_documents()

    def answer(self, question):
        context = find_relevant_passage(question, self.documents)

        if not context:
            return (
                "I could not find information related to your question in the provided articles.\n"
                "Please consult a healthcare professional."
            )

        return (
            context.strip() +
            "\n\nℹ️ This response is based strictly on published medical articles "
            "and does not replace professional medical advice."
        )