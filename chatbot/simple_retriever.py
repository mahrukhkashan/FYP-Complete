STOPWORDS = {
    "the", "is", "of", "and", "a", "in", "on", "to", "with", "for",
    "as", "by", "an", "be", "are", "from", "that", "this"
}

BANNED_PHRASES = [
    "all rights reserved",
    "published by",
    "created",
    "edition",
    "email:",
    "website:",
    "sepsis manual",
    "united kingdom sepsis trust",
    "copyright",
    "no part of this book"
]

def find_relevant_passage(question, documents):
    question_words = [
       w.lower().replace("causes", "cause").replace("causing", "cause")
       for w in question.split()
       if w.lower() not in STOPWORDS and len(w) > 3
]


    best_passage = ""
    best_score = 0

    for doc in documents:
        # Split more aggressively
        paragraphs = [
            p.strip() for p in doc.split("\n")
            if len(p.strip().split()) >= 15
        ]

        for para in paragraphs:
            lower_para = para.lower()

            # 🚫 Skip title/copyright/meta pages
            if any(bad in lower_para for bad in BANNED_PHRASES):
                continue

            # Keyword matching
            score = sum(1 for word in question_words if word in lower_para)

            # Require minimum relevance
            # Allow even 1 keyword match so answers are not missed
            if score >= 1 and score > best_score:  

                best_score = score
                best_passage = para

    return best_passage