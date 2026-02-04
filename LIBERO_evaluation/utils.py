import os
import io
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

def to_png_bytes(img_np):
    img = Image.fromarray(img_np.astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

class EpisodeParquetWriter:
    def __init__(self, root, chunk_id=0):
        self.chunk_dir = os.path.join(root, "data", f"chunk-{chunk_id:03d}")
        os.makedirs(self.chunk_dir, exist_ok=True)

        self.rows = []
        self.episode_index = 0

    def add_step(self, state, action, frame_index, task_index, timestamp):
        self.rows.append({
            "observation.state": state.tolist(),
            "action": action.tolist(),
            "timestamp": float(timestamp),
            "frame_index": int(frame_index),
            "episode_index": int(self.episode_index),
            "index": len(self.rows)-1,
            "task_index": int(task_index)
        })

    def save_episode(self):
        if len(self.rows) == 0:
            return

        df = pd.DataFrame(self.rows)
        table = pa.Table.from_pandas(df, preserve_index=False)

        filename = f"episode_{self.episode_index:06d}.parquet"
        pq.write_table(table, os.path.join(self.chunk_dir, filename))

        # clear buffer for next episode
        self.rows = []
        self.episode_index += 1
        
    def return_episode_index(self):
        return self.episode_index