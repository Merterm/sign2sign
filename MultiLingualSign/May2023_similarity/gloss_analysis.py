import spacy

# Load the English language model in spaCy
nlp = spacy.load("zh_core_web_sm")

# Define the sentence to be analyzed
sentence = "你小张什么时间认识"
# Process the sentence using spaCy's pipeline
doc = nlp(sentence)

# Get the subject, object, and verb tokens
subject = None
verb = None
object_ = None
for token in doc:
    print(token.text, token.dep_)
    if token.dep_ == "nsubj":
        subject = token
    elif token.dep_ == "dobj":
        object_ = token
    elif token.dep_ == "ROOT":
        verb = token

print(subject, object_, verb)
# Check if the sentence has an SOV or SVO order
if subject is not None and verb is not None and object_ is not None:
    if subject.i < object_.i and verb.i < object_.i:
        print("SVO")
    elif object_.i < subject.i and verb.i < object_.i:
        print("OSV")
    elif object_.i < subject.i and verb.i > object_.i:
    	print("SOV")
else:
    print("Cannot determine order")