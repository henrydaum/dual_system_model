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

from graph import KnowledgeGraph, StepInfo
from poker_monster.engine import GameEngine
from llmClass import OpenAILLM
from embedClass import SentenceTransformerEmbedder
from Thinker import Thinker

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

client = openai.OpenAI()  # Uses OPENAI_API_KEY from environment variable
OPENAI_MODEL_NAME = "gpt-4o-mini"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if __name__ == "__main__":
    # Initialize the graph and engine.
    graph = KnowledgeGraph(db_path=BASE_DIR / "graph.db")
    engine = GameEngine()
    
    models = {}
    models['llm'] = OpenAILLM(OPENAI_MODEL_NAME)
    models['embed'] = SentenceTransformerEmbedder("BAAI/bge-small-en-v1.5")
    models['llm'].load()
    models['embed'].load()

    thinker = Thinker(models['llm'], graph)

    # Reset the game and start a new sequence in the graph that corresponds to the current game.
    graph.start_new_sequence()
    engine.reset()

    agent_stepinfo = {
        "hero": StepInfo(src_agent_id="hero"),
        "monster": StepInfo(src_agent_id="monster")
    }

    while engine.get_results() is None:
        # Getting info needed to choose an action.
        src_agent = engine.gs.turn_priority  # This is the current agent.
        # Pull up the corresponding stepinfo dataclass.
        current_stepinfo = agent_stepinfo[src_agent]
        # The current node becomes the old node. The current agent becomes the old agent.
        if current_stepinfo.src_id is not None:
            current_stepinfo.dst_agent_id = src_agent
            current_stepinfo.dst_id = current_stepinfo.src_id
        # Get the basic description.
        gamestate_text, actions_text = engine.get_display_text()
        # Assign it to the dataclass.
        current_stepinfo.src_id = gamestate_text
        # Compare expectation to reality
        if current_stepinfo.dst_id is not None:
            current_stepinfo.action_verified = thinker.compare_expectation_vs_reality(current_stepinfo.dst_id, current_stepinfo.src_id, current_stepinfo.expected_results)
            # Now that all the information is gathered, record it.
            graph.record_step(current_stepinfo)

        # Display it.
        print(gamestate_text)
        print(actions_text)
        # Describe the issue at hand.
        current_stepinfo.problem_description = thinker.describe_problem(gamestate_text, actions_text)
        # Get the embedding for the problem description.
        current_stepinfo.description_embedding = models['embed'].encode([current_stepinfo.problem_description])[0]
        # Search for similar past problems in the graph along with their solutions.
        current_stepinfo.similar_steps = graph.find_similar_problems(current_stepinfo, 1)

        # Choosing an action based on info and enacting it.
        while True:
            if src_agent == "hero":
                # Find legal actions based on info.
                legal_actions = engine.get_legal_actions(actions_text)
                # Choose an action (currently a random legal one)
                action_id = random.choice(legal_actions)
                # Fake reasoning/description.
                reasoning = "Randomly chosen."
                description = "None"
            elif src_agent == "monster":
                # Use the Thinker to recommend an action.
                action_id, reasoning, expectation = thinker.recommend_action(gamestate_text, actions_text, current_stepinfo.similar_steps)
                current_stepinfo.reasoning_for_action = reasoning
                current_stepinfo.expected_results = expectation
            # Get the text of the chosen action for display and graph recording purposes.
            current_stepinfo.action_str = engine.get_action_text(actions_text, action_id)
            # Print the chosen action.
            print(f"Taking action: {current_stepinfo.action_str}")
            # Attempt to iterate the engine.
            legal, reason = engine.iterate(action_id)
            # If the action is legal, continue.
            if legal:
                break
            # If the action is illegal, print the reason and choose again.
            print(f"Illegal action chosen: {action_str}. Please try again. Reason: {reason}")
        
        # Get info for the new state after the action is taken, for graph recording purposes.
        dst_agent = engine.gs.turn_priority
        dst_gs, actions_text = engine.get_display_text()
        
        # Update the graph.
        graph.record_step(current_stepinfo)
    
    # After the game is over, fetch the rewards.
    rewards = engine.get_results()
    # Display the winner.
    print(f"WINNER: {engine.gs.winner} - {rewards}")
    # Finalize the game sequence with the rewards for each player.
    graph.finalize_sequence(rewards)

    # Learning step?