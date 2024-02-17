from transformers import pipeline

Classifier = pipeline("sentiment-analysis")
Classifier("This is fucking awesome shit")

#[{'label': 'POSITIVE', 'score': 0.9993459582328796}]

Classifier = pipeline("zero-shot-classification")
#This pipeline is called zero-shot because you don’t need to fine-tune the model on your data to use it.
Classifier("Marx is a fucking great guy in social studies", candidate_labels=["politician", "actor", "sociologist", "science"])

#{'sequence': 'Marx is a fucking great guy in social studies',
#  'labels': ['sociologist', 'actor', 'politician', 'science'], 
# 'scores': [0.9708995819091797, 0.01123867742717266,
#  0.009510486386716366, 0.008351252414286137]}

Classifier = pipeline("text-generation")
Classifier("I love using NLP in political science", max_length=50)

#[{'generated_text': 'I love using NLP in political science. I love studying how the theory of change occurs."\n\nHe had a more nuanced view about the political science movement over the last decade, where he has been involved in both political activism and political journalism.'}]


#You can use any model from ModelHub

Classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
Text = "For example, Luhmann demonstrated, the semantics of individuality underwent a profound transformation in the transition from traditional to modern society. As an effect of this structural change, an individual’s individuality was no longer defined by the affiliation to a specific societal strata (social class), but rather conceived as something to be shaped and maintained independently of former class ties"
candidate_labels = ["social class", "individuality", "affiliation", "societal strata"]
Classifier(Text, candidate_labels=candidate_labels)

#'labels': ['individuality', 'societal strata', 'social class', 'affiliation'], 'scores': [0.8231831789016724, 0.08979853242635727, 0.0644567683339119, 0.022561566904187202]}


Classifier = pipeline("ner", grouped_entities=True)
Classifier("My name is Durkheim and I live in New York City")

#{'entity_group': 'PER', 'score': 0.9915867, 'word': 'Durkheim', 'start': 11, 'end': 19},
#{'entity_group': 'LOC', 'score': 0.9992204, 'word': 'New York City', 'start': 34, 'end': 47}]

#You can also use any model from ModelHub like summarization, question-answering and translation


