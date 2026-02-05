import sqlite3
import logging
import openai
import os
from pydantic import BaseModel, Field
from typing import List, Literal
from dataclasses import dataclass
from pathlib import Path
import torch
import random
import numpy as np
import uuid

from graph import KnowledgeGraph
from poker_monster.engine import GameEngine

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

client = openai.OpenAI()  # Uses OPENAI_API_KEY from environment variable
OPENAI_MODEL_NAME = "gpt-4o-mini"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def get_embedding(text):
    return np.random.rand(10).astype(np.float32)

if __name__ == "__main__":
    graph = KnowledgeGraph(db_path=BASE_DIR / "graph.db")
    engine = GameEngine()

    graph.start_new_sequence()
    engine.reset()

    sequence_id = str(uuid.uuid4())

    while engine.get_results() is None:
        active_agent = engine.gs.turn_priority

        src_gs, actions_text = engine.get_display_text()
        print(src_gs)
        print(actions_text)
        # Testing random actions.
        while True:
            action_id = random.randint(0, engine.num_actions-2)
            legal, action_str = engine.iterate(action_id)
            if legal:
                break
        print(f"Taking action: {action_str}")
        
        dst_gs, actions_text = engine.get_display_text()

        graph.record_step(
            agent_id=active_agent,
            src_id=src_gs,
            dst_id=dst_gs,
            action_str=action_str,
            src_embedding=get_embedding(src_gs),
            dst_embedding=get_embedding(dst_gs),
            reasoning="Random Agent"
        )
    
    print(f"WINNER: {engine.gs.winner}")
    rewards = engine.get_results()
    graph.finalize_sequence(rewards)