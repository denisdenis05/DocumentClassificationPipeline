import re

from transformers import pipeline

class DocumentClassifier:
    def __init__(self, model_name: str = "facebook/bart-large-mnli"):
        self.classifier = pipeline(
            "zero-shot-classification",
            model=model_name,
        )
        self.categories = [
            "Invoice",
            "Receipt",
            "Scientific Paper",
            "Resume",
            "Contract"
        ]
        self.keyword_heuristics = {
            "Receipt": ["receipt", "invoice", "due date", "bill to", "vat number", "iban", "remittance", "gratuity", "tip",
                        "change due", "cashier", "auth code", "transaction", "transaction id","charges", "statement", "balance",
                        "total", "subtotal", "cash", "contactless", "credit card", "debit card", "quantity", "qty"],
            "Scientific Paper": ["abstract", "introduction", "conclusion", "methodology", "references", "et al.", "doi", "hypothesis", "related work"],
            "Resume": ["experience", "education", "skills", "objective", "certifications", "employment history"],
            "Contract": ["agreement", "whereas", "hereby", "governing law", "severability", "in witness whereof"]
        }

    def classify_text(self, text: str) -> str:
        if not text.strip():
            return "Unknown"

        text_lower = text.lower()
        category_scores = {category: 0 for category in self.categories}

        for category, keywords in self.keyword_heuristics.items():
            for keyword in keywords:
                if re.search(rf'\b{re.escape(keyword)}\b', text_lower):
                    category_scores[category] += 1

        sorted_scores = sorted(category_scores.items(), key=lambda item: item[1], reverse=True)

        best_category = max(category_scores, key=category_scores.get)
        highest_score = category_scores[best_category]

        second_highest_score = sorted_scores[1][1] if len(sorted_scores) > 1 else 0

        margin = highest_score - second_highest_score

        if highest_score >= 2 and margin >= 2:
            return best_category

        result = self.classifier(text[:1000], self.categories, multi_label=False)
        return result['labels'][0]