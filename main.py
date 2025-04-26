from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def clean_sentence(sentence):
    return sentence.split("+++$+++")[4].strip(" ").replace("\n", "")


sentences = []

with open("movie_lines.txt") as f:
  for i in f.readlines():
      sentences.append(clean_sentence(i))




vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(sentences)

def chatbot_response(user_input):
    user_vec = vectorizer.transform([user_input])
    similarities = cosine_similarity(user_vec, X)
    best_match = similarities.argmax()
    return sentences[best_match]


while True:
    user = input("You: ")
    if user.lower() == 'bye':
        print("Bot: Goodbye!")
        break
    response = chatbot_response(user)
    print(f"Bot: {response}")




