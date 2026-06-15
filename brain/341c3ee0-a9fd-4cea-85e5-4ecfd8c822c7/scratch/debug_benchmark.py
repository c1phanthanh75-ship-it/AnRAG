import sys
from pathlib import Path
import json

# Add project root to sys.path
sys.path.append(r"c:\Users\Admin\Documents\AnRAG")

from datasets import load_dataset
from anrag.benchmark_gt import load_hotpotqa, resolve_gold_official
from anrag.config import Settings
from anrag.store import SQLiteTreeStore
from anrag.pipeline import ingest_blocks
from anrag.chunking import fixed_chunking

def main():
    tmp_dir = Path(r"c:\Users\Admin\Documents\AnRAG\scratch_tmp")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    
    # Load 1 HotpotQA sample
    print("Loading HotpotQA...")
    hotpot = list(load_dataset("hotpotqa/hotpot_qa", name="distractor", split="validation"))
    sample = [hotpot[0]]
    
    hotpot_path = tmp_dir / "hotpotqa_debug.jsonl"
    with open(hotpot_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(sample[0]) + "\n")
        
    print("Loading benchmark format...")
    documents, questions = load_hotpotqa(hotpot_path)
    
    doc_id = next(iter(documents))
    blocks = documents[doc_id]
    
    print(f"\nTotal blocks: {len(blocks)}")
    
    # Run fixed chunking directly
    from anrag.text import chunk_text_by_tokens
    text = "\n\n".join(block.text for block in blocks if block.text.strip())
    parts = chunk_text_by_tokens(text, 260, 40)
    
    print(f"Total parts (chunks): {len(parts)}")
    part = parts[0]
    print(f"\nFirst chunk text (len={len(part)}):\n{part[:500]}...\n")
    
    for i, b in enumerate(blocks):
        b_text_norm = " ".join(b.text.split())
        matched = b_text_norm in part
        print(f"Block {i} (kind={b.kind}, metadata={b.metadata}):")
        print(f"  Norm Text: {b_text_norm[:100]}...")
        print(f"  Matched in first chunk? {matched}")
        if not matched and i < 5:
            # Let's check why it didn't match
            print(f"  Looking for: {b_text_norm}")
            
if __name__ == "__main__":
    main()
