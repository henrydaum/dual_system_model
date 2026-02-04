import sqlite3
import logging
import openai
import os
from pydantic import BaseModel, Field
from typing import List, Literal
from dataclasses import dataclass
from pypdf import PdfReader
from pathlib import Path
import torch

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

client = openai.OpenAI()  # Uses OPENAI_API_KEY from environment variable
OPENAI_MODEL_NAME = "gpt-4o-mini"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 1. Find every object and process in the document. With the entire document as context, ask an LLM to find them all.
# 2. Deduplicate the objects and processes found in step 1.
# 3. For each unique object and process, find its relationships to other objects and processes and find its properties. With the entire document as context, ask an LLM about one object or process at a time.

@dataclass
class Triple:
    head: str
    connection: str
    tail: str
    feeling: float = 0.0
    context: str = None

class Entity(BaseModel):
    name: str = Field(..., description="The precise name (e.g., 'Encyclopedia Britannica').")
    type: Literal["Object", "Process"] = Field(..., description="Is it a static object or a dynamic process?")
    category: Literal[
        "Consciousness", 
        "Mental Factor", 
        "Materiality", 
        "Cognitive Process", 
        "Conditionality", 
        "Society", 
        "Evolution", 
        "Freedom"
    ] = Field(..., description="A general category for this entity.")

class Entities(BaseModel):
    entities: List[Entity]

def extract_entities(document_text):
    messages=[
            {"role": "system", "content": f"Here is the context text:\n{document_text}"},
            {"role": "user", "content": f"Extract all objects and processes."}
        ]
    response = client.responses.parse(model=OPENAI_MODEL_NAME, input=messages, text_format=Entities)
    entities = response.output_parsed
    print(f"Extracted entities: {[e.name for e in entities.entities]}")
    return entities

class Property(BaseModel):
    property: str = Field(..., description="The descriptive property (e.g. 'slow', 'can dance', 'going 40mph').")
    # context: str = Field(..., description="A short quote from the text verifying this property.")

class Relationship(BaseModel):
    target_entity: str = Field(..., description="The exact name of the OTHER entity this connects to.")
    relationship: str = Field(..., description="The precise relationship (e.g., 'describes', 'becomes', 'laughs at').")
    category: Literal[
        "causes",
        "controls",
        "part_of",
        "is_a",
        "precedes",
        "transforms_into",
        "opposes"
    ] = Field(..., description="A general category for this relationship.")
    feeling: float = Field(
        ..., 
        ge=-1.0, 
        le=1.0, 
        description="The overall feeling evoked from -1.0 (Unpleasant) to 1.0 (Pleasant), with 0 being neutral."
    )
    # context: str = Field(..., description="A short quote from the text verifying this connection.")

class Connections(BaseModel):
    properties: List[Property]
    relationships: List[Relationship]

def extract_relationships_and_properties(entity_list, document_text):
    all_properties = []
    all_relationships = []
    for entity in entity_list:
        print(f"Processing entity: {entity.name}")
        # Use prompt caching so that the start of the document is not sent to the LLM every time, saving tokens.
        messages=[
                {"role": "system", "content": f"Here is the context text:\n{document_text}"},
                {"role": "user", "content": f"Extract relationships for: '{entity.name}'"}
            ]
        try:
            response = client.responses.parse(model=OPENAI_MODEL_NAME, input=messages, text_format=Connections)
            connections = response.output_parsed

            for p in connections.properties:
                p_dict = p.model_dump()       # Convert to dict: {'property': 'fast', ...}
                p_dict['source'] = entity.name
                all_properties.append(p_dict)

            for r in connections.relationships:
                r_dict = r.model_dump()       # Convert to dict
                r_dict['source'] = entity.name
                all_relationships.append(r_dict)
        except Exception as e:
            logger.error(f"Failed to process {entity.name}: {e}")
            continue

    return all_properties, all_relationships

if __name__ == "__main__":
    THE_WHOLE = BASE_DIR / "The Whole by Henry Daum.pdf"
    document_text = get_text_from_pdf(THE_WHOLE)
    document_text = document_text[:1000]

    entities = extract_entities(document_text)
    # Missing deduplication step
    properties, relationships = extract_relationships_and_properties(entities.entities, document_text)

    triples_list = []
    for p in properties:
        triples_list.append(Triple(head=p['source'], connection='has_property', tail=p['property'], context=None))
    for r in relationships:
        triples_list.append(Triple(head=r['source'], connection=r['relationship'], tail=r['target_entity'], feeling=r['feeling'], context=None))
    
    for t in triples_list:
        print(f"{t.head} -- {t.connection} --> {t.tail} (Feeling: {t.feeling})")