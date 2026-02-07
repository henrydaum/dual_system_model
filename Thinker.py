from pydantic import BaseModel
import json

GAME_RULES = """
HOW TO PLAY POKER MONSTER
Object/ways to win: Each player starts with 15 health, and a 20-card deck. The first player to run out of health or cards in their deck loses the game.

To start the game: Each player chooses the deck they want to play (red = Monster, green = Hero). Flip a coin to see who goes first. Both players shuffle their decks and draw 4 cards. Finally, the person going first starts.

Turn order: This is a turn-based game, and each turn has the same pattern. At the start of the turn, the player whose turn it is draws a card. Then, if they have any power cards, then they gain 1 power for each one they have (this is skipped on turn 1). From there, they can play the cards in their hand using the set of rules explained on the other side of this card. At the end of the turn, the player whose turn it is passes their turn to their opponent; and they start theirs. This pattern repeats until somebody wins the game.

Note: The player going first doesn't draw a card on their initial turn.

Card rules: Any card can be played face-up or face-down. Face-down cards are known as power cards, and these give the power needed to play cards face-up. In this way, every card has a dual usage. To play a power card, simply take any card from your hand and place it onto the board face-down. When you do this, you have made a power card. Face-up cards are different. While power cards are hidden from your opponent, face-up cards are revealed as you play them. To do this requires power, which is explained below. To play a card from your hand in the face-up configuration, simply pay its power cost, then follow its text. Each card basically does what it says, except short cards are discarded after use, while long cards remain on the board until killed (their health is at the bottom), and then are discarded.

You can (and should, in the early turns of the game) play 1 power card per turn, but it's not required. You can play any number of face-up cards you have power for.

Power: The power cost to play a face-up card is in the top-right corner of the card. Power is given by power cards; when you play a power card, you get 1 power, and when your turn starts, you get 1 power for each power card you control. Power resets to 0 when the turn ends.

Shake hands at the end. Good luck!
"""

class ProblemDescription(BaseModel):
    problem_description: str

class Recommendation(BaseModel):
    recommended_action_id: int
    reasoning_for_action: str
    expected_results: str

class ActualResults(BaseModel):
    action_verified: bool

class Thinker:
    def __init__(self, llm, graph):
        self.llm = llm
        self.graph = graph

    def describe_problem(self, gamestate_text, actions_text):
        prompt = f"You are playing a game called Poker Monster. Here are the rules:\n{GAME_RULES}\n\nHere is the current game state:\n{gamestate_text}\n\nHere are your available actions:\n{actions_text}\nBased on the current game state and available actions, make a description of the problem you are facing. Don't focus on solutions, just describe the issue at hand.\n"
        response = self.llm.invoke(prompt, response_format=ProblemDescription)
        response = json.loads(response)
        return response['problem_description']

    def recommend_action(self, gamestate_text, actions_text, similar_steps=None):
        prompt = f"You are playing a game called Poker Monster. Here are the rules:\n{GAME_RULES}\n\nHere is the current game state:\n{gamestate_text}\n\nHere are the possible actions you can take, with their corresponding IDs:\n{actions_text}\n\nBased on the current game state and the possible actions, make a recommendation for the next action and explain your reasoning for such action. Finally, explain what you think will happen as a result of taking that action.\n\n"
        if similar_steps:
            prompt += "Here are some similar past situations you have encountered along with their solutions that might help inform your decision:\n"
            for i, step in enumerate(similar_steps):
                prompt += f"\nMemory [{i+1}] - What happened: {step['problem_description']} | Solution: {step['reasoning_for_action']}\n"
        response = self.llm.invoke(prompt, response_format=Recommendation)
        response = json.loads(response)
        return response['recommended_action_id'], response['reasoning_for_action'], response['expected_results']

    def compare_expectation_vs_reality(self, current_state, previous_state, expected_results):
        prompt = f"You are playing a game called Poker Monster. Here are the rules:\n{GAME_RULES}\n\nHere is was the previous game state:\n{previous_state}\nHere were your expectations for the results of your last action:\n{expected_results}\n\nHere are the actual results that occurred after taking that action:\n{current_state}\n\nBased on this, analyze whether your expectations matched reality. This information will be used to improve your future decision-making. Respond with whether the action was verified (true/false).\n"
        response = self.llm.invoke(prompt, response_format=ActualResults)
        response = json.loads(response)
        return response['action_verified']