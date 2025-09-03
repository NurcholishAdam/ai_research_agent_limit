# entity_linker.py
import spacy
from fuzzywuzzy import process

class EntityLinker:
    def __init__(self, graph_nodes):
        self.nlp = spacy.load("en_core_web_sm")
        self.graph_nodes = graph_nodes  # List of known entities in graph

    def extract_entities(self, query):
        doc = self.nlp(query)
        ents = [ent.text for ent in doc.ents]
        return ents

    def link_entities(self, ents):
        linked = {}
        for ent in ents:
            match, score = process.extractOne(ent, self.graph_nodes)
            if score > 80:
                linked[ent] = match
        return linked

query = "Who likes apples?"
graph_nodes = ["Andrew", "Buddy", "Scout", "apples", "bananas"]
linker = EntityLinker(graph_nodes)
entities = linker.extract_entities(query)
linked = linker.link_entities(entities)
print(linked)  # {'apples': 'apples'}
