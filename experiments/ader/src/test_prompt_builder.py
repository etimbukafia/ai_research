#!/usr/bin/env python3
"""Quick test of the prompt builder."""

from uuid import UUID
from pathlib import Path
from experiments.ader.src.db import create_default_db
from experiments.ader.src.prompt_builder import build_ader_prompt


def test_prompt_builder():
    print("Creating default DB...")
    db = create_default_db()
    
    alex_id = UUID("d3b07384-d113-4956-a5e2-4c5b3648a301")
    
    print(f"\nFetching memory for Alex...")
    mem = db.get_ader_memory(alex_id)
    
    print(f"\nBuilding prompt for Alex (Regulated Profile)...")
    prompts_dir = Path(__file__).parent / "prompts"
    final_prompt = build_ader_prompt(mem, "Alex (Regulated Profile)", prompts_dir=prompts_dir)
    
    print("\n" + "="*80)
    print("RENDERED PROMPT:")
    print("="*80 + "\n")
    print(final_prompt)
    print("\n" + "="*80)
    
    # Also test with Taylor (overloaded profile)
    taylor_id = UUID("e6c98522-83b6-4bfe-bb4f-b3a1a6b0cfa0")
    print(f"\n\nFetching memory for Taylor...")
    mem_taylor = db.get_ader_memory(taylor_id)
    
    print(f"\nBuilding prompt for Taylor (Overloaded Profile)...")
    final_prompt_taylor = build_ader_prompt(mem_taylor, "Taylor (Overloaded Profile)", prompts_dir=prompts_dir)
    
    print("\n" + "="*80)
    print("RENDERED PROMPT FOR TAYLOR:")
    print("="*80 + "\n")
    print(final_prompt_taylor[:1000])  # First 1000 chars
    print("\n... (truncated for brevity)")
    

if __name__ == "__main__":
    test_prompt_builder()
