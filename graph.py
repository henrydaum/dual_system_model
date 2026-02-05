import sqlite3
import uuid
import numpy as np

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
                outcome REAL DEFAULT 0.0, 
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
                reasoning TEXT,
                FOREIGN KEY(sequence_id) REFERENCES sequences(sequence_id)
                FOREIGN KEY(src_id) REFERENCES nodes(id),
                FOREIGN KEY(dst_id) REFERENCES nodes(id)
            )
        """)

        # 3. NODES (States)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS nodes (
                id TEXT PRIMARY KEY,
                embedding BLOB,
                agent_id TEXT,
                total_reward REAL DEFAULT 0.0,
                count INTEGER DEFAULT 0,
                avg_reward REAL DEFAULT 0.0
            )
        """)

        # 4. EDGES (Actions)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS edges (
                src_id TEXT,
                dst_id TEXT,
                action_str TEXT,
                agent_id TEXT,
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

    def record_step(self, agent_id, src_id, src_embedding, action_str, dst_id, dst_embedding, reasoning=""):
        self._ensure_node(src_id, agent_id, src_embedding)
        self._ensure_node(dst_id, agent_id, dst_embedding)
        self._ensure_edge(src_id, dst_id, action_str, agent_id)

        self.conn.execute("""
            INSERT INTO steps (sequence_id, step_num, src_id, dst_id, action_str, reasoning)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (self.current_sequence_id, self.step_counter, src_id, dst_id, action_str, reasoning))
        
        self.step_counter += 1
        self.conn.commit()

    def _ensure_node(self, node_id, agent_id, embedding):
        blob = embedding.tobytes() if embedding is not None else None
        self.conn.execute("""
            INSERT OR IGNORE INTO nodes (id, agent_id, embedding)
            VALUES (?, ?, ?)
        """, (node_id, agent_id, blob))

    def _ensure_edge(self, src, dst, act, agent):
        self.conn.execute("""
            INSERT OR IGNORE INTO edges (src_id, dst_id, action_str, agent_id)
            VALUES (?, ?, ?, ?)
        """, (src, dst, act, agent))

    # --- LEARNING ---

    def finalize_sequence(self, rewards_dict):
        """
        Updates stats based on results for Nodes and Edges touched in this sequence, filtered by the Agent ID.
        """
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

    