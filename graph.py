import sqlite3
import uuid
import numpy as np
from dataclasses import dataclass
from typing import List, Any

@dataclass
class StepInfo:
    src_agent_id: str = None
    dst_agent_id: str = None
    src_id: str = None
    dst_id: str = None
    action_str: str = None
    problem_description: str = None
    description_embedding: Any = None 
    reasoning_for_action: str = None
    expected_results: str = None
    action_verified: bool = False
    similar_steps: List[dict] = None

class KnowledgeGraph:
    def __init__(self, db_path="brain.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row 
        self._init_schema()
        
        # Context
        self.current_sequence_id = None
        self.step_counter = 0

    def _init_schema(self):
        cur = self.conn.cursor()
        
        # 1. SEQUENCES (The Chain)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS sequences (
                sequence_id TEXT PRIMARY KEY,
                outcome TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # 2. STEPS (The Chain Links)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS steps (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sequence_id TEXT,
                step_num INTEGER,
                src_id TEXT,
                dst_id TEXT,
                action_str TEXT,
                src_agent_id TEXT,
                problem_description TEXT,
                description_embedding BLOB,
                reasoning_for_action TEXT,
                expected_results TEXT,
                action_verified BOOLEAN,
                FOREIGN KEY(sequence_id) REFERENCES sequences(sequence_id)
                FOREIGN KEY(src_id) REFERENCES nodes(id),
                FOREIGN KEY(dst_id) REFERENCES nodes(id)
            )
        """)

        # 3. NODES (States)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS nodes (
                agent_id TEXT,
                id TEXT PRIMARY KEY,
                total_reward REAL DEFAULT 0.0,
                count INTEGER DEFAULT 0,
                avg_reward REAL DEFAULT 0.0
            )
        """)

        # 4. EDGES (Actions)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS edges (
                agent_id TEXT,
                src_id TEXT,
                dst_id TEXT,
                action_str TEXT,
                total_reward REAL DEFAULT 0.0,
                count INTEGER DEFAULT 0,
                avg_reward REAL DEFAULT 0.0,
                PRIMARY KEY (src_id, dst_id, action_str),
                FOREIGN KEY(src_id) REFERENCES nodes(id),
                FOREIGN KEY(dst_id) REFERENCES nodes(id)
            )
        """)
        self.conn.commit()

    # --- RECORDING ---

    def start_new_sequence(self):
        self.current_sequence_id = str(uuid.uuid4())
        self.step_counter = 0
        self.conn.execute("INSERT INTO sequences (sequence_id) VALUES (?)", (self.current_sequence_id,))
        self.conn.commit()

    def record_step(self, step: StepInfo):
        self._create_node(step.src_id, step.src_agent_id)
        self._create_node(step.dst_id, step.dst_agent_id)
        self._create_edge(step.src_id, step.dst_id, step.action_str, step.src_agent_id)  # Edges are made by the source agent.

        blob = step.description_embedding.tobytes() if step.description_embedding is not None else None

        self.conn.execute("""
            INSERT INTO steps (sequence_id, step_num, src_id, dst_id, action_str, src_agent_id, problem_description, description_embedding, reasoning_for_action, expected_results, action_verified)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (self.current_sequence_id, self.step_counter, step.src_id, step.dst_id, step.action_str, step.src_agent_id, step.problem_description, blob, step.reasoning_for_action, step.expected_results, step.action_verified))
        
        self.step_counter += 1
        self.conn.commit()

    def _create_node(self, node_id, agent_id):
        self.conn.execute("""
            INSERT OR IGNORE INTO nodes (id, agent_id)
            VALUES (?, ?)
        """, (node_id, agent_id))

    def _create_edge(self, src, dst, act, agent_id):
        self.conn.execute("""
            INSERT OR IGNORE INTO edges (src_id, dst_id, action_str, agent_id)
            VALUES (?, ?, ?, ?)
        """, (src, dst, act, agent_id))
    # --- LEARNING ---

    def finalize_sequence(self, rewards_dict):
        """
        Updates stats based on results for Nodes and Edges touched in this sequence, filtered by the Agent ID.
        """
        self.conn.execute("""
            UPDATE sequences
            SET outcome = ?
            WHERE sequence_id = ?
        """, (str(rewards_dict), self.current_sequence_id))

        # Iterate over the rewards (e.g., hero: 1.0, monster: -1.0)
        for agent_id, reward in rewards_dict.items():
            
            # 1. Update NODES for this agent involved in this sequence
            self.conn.execute("""
                UPDATE nodes
                SET count = count + 1,
                    total_reward = total_reward + ?,
                    avg_reward = (total_reward + ?) / (count + 1)
                WHERE agent_id = ?
                  AND id IN (
                      SELECT src_id FROM steps WHERE sequence_id = ?
                      UNION
                      SELECT dst_id FROM steps WHERE sequence_id = ?
                  )
            """, (reward, reward, agent_id, self.current_sequence_id, self.current_sequence_id))

            # 2. Update EDGES for this agent involved in this sequence
            self.conn.execute("""
                UPDATE edges
                SET count = count + 1,
                    total_reward = total_reward + ?,
                    avg_reward = (total_reward + ?) / (count + 1)
                WHERE agent_id = ?
                  AND (src_id, dst_id, action_str) IN (
                      SELECT src_id, dst_id, action_str FROM steps WHERE sequence_id = ?
                  )
            """, (reward, reward, agent_id, self.current_sequence_id))

        self.conn.commit()

    # --- ANALYTICS ---

    def find_similar_problems(self, step: StepInfo, limit=5):
        cur = self.conn.cursor()
        # Fetch ID as well so we can track lineage if needed
        cur.execute("""
            SELECT problem_description, description_embedding, reasoning_for_action 
            FROM steps 
            WHERE description_embedding IS NOT NULL 
            AND action_verified = 1
            AND src_agent_id = ?
        """, (step.src_agent_id, ))
        rows = cur.fetchall()

        if not rows: return []

        valid_data = [] # Store metadata here
        valid_vectors = [] # Store vectors here for numpy

        for desc, blob, reasoning in rows:
            if blob:
                vec = np.frombuffer(blob, dtype=np.float32)
                # MANUAL NORMALIZATION (L2) to ensure Dot Product == Cosine Sim
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm
                
                valid_vectors.append(vec)
                valid_data.append({"problem_description": desc, "reasoning_for_action": reasoning})

        if not valid_vectors:
            return []

        emb_matrix = np.vstack(valid_vectors)
        
        # Normalize the query vector too
        query_norm = np.linalg.norm(step.description_embedding)
        if query_norm > 0:
            description_embedding_normalized = step.description_embedding / query_norm

        scores = np.dot(emb_matrix, description_embedding_normalized)

        # Get top K indices
        k = min(limit, len(scores))
        top_indices = np.argsort(scores)[-k:][::-1]

        results = []
        for idx in top_indices:
            # Use the index to grab the data from the list
            results.append(valid_data[idx])
            
        return results