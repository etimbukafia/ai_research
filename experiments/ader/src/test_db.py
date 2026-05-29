from uuid import UUID, uuid4
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
    new_session_id = uuid4()
    memory.working.last_session_id = new_session_id
    if memory.episodic:
        new_episode = memory.episodic[0].model_copy(update={
            "event": "completed a new planning review",
            "trigger": "reflected on next week",
            "response": "created a fresh action list",
            "outcome": "felt more in control"
        })
    else:
        new_episode = None
    if new_episode is not None:
        memory.episodic.append(new_episode)
    db.save_ader_memory(alex_id, memory)
    
    print("\nRe-stitching AderMemory from db...")
    updated_memory = db.get_ader_memory(alex_id)
    
    # Assertions to ensure data was roundtripped correctly
    assert updated_memory.working.active_goals_completed["finish science project"] is True
    assert updated_memory.affective.stress_level == 0.1
    assert db.working_memories[0].last_session_id == new_session_id
    assert db.semantic_memories[0].last_session_id == new_session_id
    assert db.procedural_memories[0].last_session_id == new_session_id
    assert db.affective_states[0].last_session_id == new_session_id
    print("Success: Memory successfully deconstructed, saved, and reconstructed!")

if __name__ == "__main__":
    test_mock_db()
