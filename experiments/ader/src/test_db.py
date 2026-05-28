from uuid import UUID
from experiments.ader.src.db import create_default_db

def test_mock_db():
    print("Initializing mock database...")
    db = create_default_db()
    
    print(f"Total Users: {len(db.users)}")
    for user in db.users:
        print(f" - {user.name} ({user.user_id})")

    # Pick Alex's ID
    alex_id = UUID("d3b07384-d113-4956-a5e2-4c5b3648a301")
    
    print("\nStitching AderMemory for Alex...")
    memory = db.get_ader_memory(alex_id)
    
    print("\nStitched AderMemory content summary:")
    print(memory.render())
    
    # Modify memory state and verify saving
    print("\nModifying Alex's memory (completing task 'finish science project' & setting stress to 0.1)...")
    memory.working.active_goals_completed["finish science project"] = True
    memory.affective.stress_level = 0.1
    
    db.save_ader_memory(alex_id, memory)
    
    print("\nRe-stitching AderMemory from db...")
    updated_memory = db.get_ader_memory(alex_id)
    
    # Assertions to ensure data was roundtripped correctly
    assert updated_memory.working.active_goals_completed["finish science project"] is True
    assert updated_memory.affective.stress_level == 0.1
    print("Success: Memory successfully deconstructed, saved, and reconstructed!")

if __name__ == "__main__":
    test_mock_db()
