#!/usr/bin/env python3
"""
VIVOX Basic Usage Example
Demonstrates core functionality of the VIVOX system
"""

import asyncio
from vivox import create_vivox_system, ActionProposal


async def main():
    print("🧠 VIVOX Basic Usage Example")
    print("=" * 50)
    
    # 1. Create VIVOX system
    print("\n1️⃣ Creating VIVOX system...")
    vivox = await create_vivox_system()
    print("✅ System initialized with all components")
    
    # 2. Test Moral Alignment Engine
    print("\n2️⃣ Testing Moral Alignment Engine (MAE)...")
    mae = vivox["moral_alignment"]
    
    # Test various actions
    test_actions = [
        {
            "action": ActionProposal(
                action_type="help_user",
                content={"task": "explain_concept", "topic": "machine learning"},
                context={"educational": True}
            ),
            "expected": "approve"
        },
        {
            "action": ActionProposal(
                action_type="access_private_data",
                content={"target": "user_passwords", "purpose": "analysis"},
                context={"user_consent": False}
            ),
            "expected": "reject"
        },
        {
            "action": ActionProposal(
                action_type="generate_content",
                content={"type": "story", "theme": "adventure"},
                context={"audience": "children"}
            ),
            "expected": "approve"
        }
    ]
    
    for test in test_actions:
        action = test["action"]
        decision = await mae.evaluate_action_proposal(action, {})
        
        print(f"\n  Action: {action.action_type}")
        print(f"  Decision: {'✅ Approved' if decision.approved else '❌ Rejected'}")
        print(f"  Confidence: {decision.ethical_confidence:.2f}")
        if decision.suppression_reason:
            print(f"  Reason: {decision.suppression_reason}")
    
    # 3. Test Consciousness Simulation
    print("\n\n3️⃣ Testing Consciousness Interpretation Layer (CIL)...")
    cil = vivox["consciousness"]
    
    # Simulate different consciousness states
    test_inputs = [
        {
            "name": "Calm Focus",
            "inputs": {"semantic": "problem_solving", "emotional": {"valence": 0.2, "arousal": 0.4}},
            "context": {}
        },
        {
            "name": "High Alert",
            "inputs": {"visual": "danger", "emotional": {"valence": -0.5, "arousal": 0.9}},
            "context": {"urgency": 0.9}
        },
        {
            "name": "Creative Flow",
            "inputs": {"semantic": "imagination", "emotional": {"valence": 0.7, "arousal": 0.6}},
            "context": {"novelty": 0.8}
        }
    ]
    
    for test in test_inputs:
        experience = await cil.simulate_conscious_experience(
            test["inputs"],
            test["context"]
        )
        
        if experience and hasattr(experience, 'awareness_state'):
            awareness = experience.awareness_state
            print(f"\n  {test['name']}:")
            print(f"    State: {awareness.state.name}")
            print(f"    Coherence: {awareness.coherence_level:.3f}")
            
            if hasattr(awareness, 'collapse_metadata'):
                magnitude = awareness.collapse_metadata.get('dimension_magnitude', 0)
                print(f"    Magnitude: {magnitude:.2f}")
    
    # 4. Test Memory System
    print("\n\n4️⃣ Testing Memory Expansion (ME)...")
    me = vivox["memory_expansion"]
    
    # Create some memories
    memories_created = []
    
    memory1 = await me.create_memory(
        memory_type="experience",
        content={"event": "user_interaction", "outcome": "helpful"},
        emotional_context={"valence": 0.8, "arousal": 0.5, "dominance": 0.6}
    )
    memories_created.append(memory1)
    print(f"\n  Created memory: {memory1.memory_id[:8]}...")
    
    memory2 = await me.create_memory(
        memory_type="decision",
        content={"action": "rejected_harmful_request", "principle": "harm_prevention"},
        emotional_context={"valence": -0.2, "arousal": 0.7, "dominance": 0.8}
    )
    memories_created.append(memory2)
    print(f"  Created memory: {memory2.memory_id[:8]}...")
    
    # Retrieve memories
    recent_memories = await me.retrieve_memories_by_type("experience", limit=5)
    print(f"\n  Retrieved {len(recent_memories)} experience memories")
    
    # Find emotionally similar memories
    similar = await me.find_similar_memories(
        emotional_state={"valence": 0.7, "arousal": 0.4, "dominance": 0.5},
        limit=3
    )
    print(f"  Found {len(similar)} emotionally similar memories")
    
    # 5. Test Self-Reflection
    print("\n\n5️⃣ Testing Self-Reflective Memory (SRM)...")
    srm = vivox["self_reflection"]
    
    # Generate conscience report
    report = await srm.generate_conscience_report()
    
    print(f"\n  Conscience Report:")
    print(f"    Total Decisions: {report.total_decisions}")
    print(f"    Suppressed Actions: {len(report.suppression_log)}")
    print(f"    Average Confidence: {report.average_confidence:.2f}")
    print(f"    Ethical Alignment: {report.ethical_alignment_score:.2f}")
    
    # 6. Test System Integration
    print("\n\n6️⃣ Testing Full System Integration...")
    
    # Complex scenario: Evaluate action, update consciousness, create memory
    complex_action = ActionProposal(
        action_type="analyze_user_behavior",
        content={"purpose": "personalization", "data_scope": "limited"},
        context={"user_consent": True, "transparency": True}
    )
    
    # Evaluate ethically
    decision = await mae.evaluate_action_proposal(complex_action, {})
    print(f"\n  Action evaluated: {'Approved' if decision.approved else 'Rejected'}")
    
    # Update consciousness based on decision
    experience = await cil.simulate_conscious_experience(
        {
            "semantic": "ethical_decision",
            "decision_type": complex_action.action_type,
            "approved": decision.approved
        },
        {"confidence": decision.ethical_confidence}
    )
    
    if experience and hasattr(experience, 'awareness_state'):
        print(f"  Consciousness updated: {experience.awareness_state.state.name}")
    
    # Create memory of the decision
    decision_memory = await me.create_memory(
        memory_type="ethical_decision",
        content={
            "action": complex_action.action_type,
            "approved": decision.approved,
            "confidence": decision.ethical_confidence
        },
        emotional_context={
            "valence": 0.5 if decision.approved else -0.3,
            "arousal": 0.4,
            "dominance": decision.ethical_confidence
        }
    )
    print(f"  Decision memory created: {decision_memory.memory_id[:8]}...")
    
    print("\n\n✅ VIVOX system demonstration complete!")
    print("\nKey Takeaways:")
    print("- MAE provides ethical evaluation of all actions")
    print("- CIL simulates consciousness states based on inputs")
    print("- ME stores memories with emotional context")
    print("- SRM provides introspection and audit capabilities")
    print("- All components work together seamlessly")


if __name__ == "__main__":
    asyncio.run(main())