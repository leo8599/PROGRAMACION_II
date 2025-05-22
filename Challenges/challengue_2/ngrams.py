import spacy
from collections import Counter

# Load the Spanish language model
nlp = spacy.load('es_core_news_sm')

def compute_ngram_distribution(text, n):
    # Process the text with spaCy
    doc = nlp(text)
    # Tokenize the text and extract tokens
    tokens = [token.text for token in doc]
    # Generate n-grams
    ngrams_list = [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
    # Compute the frequency distribution of n-grams
    ngram_freq = Counter(ngrams_list)
    # Calculate probabilities
    total_ngrams = sum(ngram_freq.values())
    ngram_probs = {ngram: freq / total_ngrams for ngram, freq in ngram_freq.items()}
    return ngram_probs

# Example usage
text = "Los perros están ladrando fuerte afuera. Estoy corriendo en el parque."
n = 4  # Bi-grams
ngram_probs = compute_ngram_distribution(text, n)
print(f"{n}-gram Distribution:")
for ngram, prob in ngram_probs.items():
    print(f"{ngram}: {prob:.4f}")