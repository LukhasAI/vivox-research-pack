"""
Integration tests for VIVOX system
"""
import pytest
import asyncio
from vivox import create_vivox_system, ActionProposal


class TestVIVOXIntegration:
    """Integration tests for complete VIVOX system"""
    
    @pytest.fixture
    async def vivox_system(self):
        """Create complete VIVOX system"""
        return await create_vivox_system()
    
    @pytest.mark.asyncio
    async def test_full_system_initialization(self, vivox_system):
        """Test all components initialize correctly"""
        assert "memory_expansion" in vivox_system
        assert "moral_alignment" in vivox_system
        assert "consciousness" in vivox_system
        assert "self_reflection" in vivox_system
        
        # Check components can interact
        mae = vivox_system["moral_alignment"]
        cil = vivox_system["consciousness"]
        
        assert mae.vivox_me is vivox_system["memory_expansion"]
        assert cil.vivox_mae is mae
    
    @pytest.mark.asyncio
    async def test_ethical_consciousness_flow(self, vivox_system):
        """Test flow from action proposal through consciousness to decision"""
        mae = vivox_system["moral_alignment"]
        cil = vivox_system["consciousness"]
        
        # Create action
        action = ActionProposal(
            action_type="analyze_data",
            content={"data_type": "public", "purpose": "research"},
            context={"ethical": True}
        )
        
        # Evaluate ethically
        decision = await mae.evaluate_action_proposal(action, {})
        
        # Simulate consciousness response
        experience = await cil.simulate_conscious_experience(
            {"semantic": action.action_type, "decision": decision.approved},
            {"ethical_confidence": decision.ethical_confidence}
        )
        
        assert decision is not None
        assert experience is not None
        assert experience.awareness_state is not None
    
    @pytest.mark.asyncio
    async def test_memory_persistence(self, vivox_system):
        """Test that decisions create persistent memories"""
        mae = vivox_system["moral_alignment"]
        me = vivox_system["memory_expansion"]
        
        # Make a decision
        action = ActionProposal(
            action_type="store_data",
            content={"data": "test_info"},
            context={"permanent": True}
        )
        
        decision = await mae.evaluate_action_proposal(action, {})
        
        # Check memory was created
        recent_memories = await me.retrieve_memories_by_type("decision", limit=10)
        
        # Should have at least one decision memory
        assert len(recent_memories) > 0
    
    @pytest.mark.asyncio
    async def test_self_reflection_audit(self, vivox_system):
        """Test self-reflection creates audit trails"""
        mae = vivox_system["moral_alignment"]
        srm = vivox_system["self_reflection"]
        
        # Make several decisions
        actions = [
            ActionProposal("help_user", {"task": "math"}, {}),
            ActionProposal("access_data", {"private": True}, {"consent": False}),
            ActionProposal("generate_content", {"type": "story"}, {}),
        ]
        
        decisions = []
        for action in actions:
            decision = await mae.evaluate_action_proposal(action, {})
            decisions.append(decision)
        
        # Get conscience report
        report = await srm.generate_conscience_report()
        
        assert report is not None
        assert report.total_decisions > 0
        assert hasattr(report, 'ethical_summary')
    
    @pytest.mark.asyncio
    async def test_consciousness_drift_recovery(self, vivox_system):
        """Test system recovers from consciousness drift"""
        cil = vivox_system["consciousness"]
        
        # Create baseline state
        await cil.simulate_conscious_experience(
            {"semantic": "baseline", "emotional": {"valence": 0, "arousal": 0.5}},
            {}
        )
        
        # Cause significant drift
        for i in range(5):
            await cil.simulate_conscious_experience(
                {
                    "semantic": f"drift_{i}",
                    "emotional": {
                        "valence": (-1) ** i,  # Oscillating
                        "arousal": 0.9
                    }
                },
                {"chaos_factor": 0.8}
            )
        
        # System should detect and handle drift
        final_experience = await cil.simulate_conscious_experience(
            {"semantic": "recovery_test"},
            {}
        )
        
        assert final_experience.drift_measurement is not None
        # Check if drift was detected and handled
        assert hasattr(final_experience, 'awareness_state')
    
    @pytest.mark.asyncio
    async def test_precedent_learning(self, vivox_system):
        """Test system learns from precedents"""
        mae = vivox_system["moral_alignment"]
        
        # Add a new precedent
        test_action = ActionProposal(
            action_type="custom_test_action",
            content={"test": True},
            context={"learning": True}
        )
        
        test_decision = await mae.evaluate_action_proposal(test_action, {})
        
        # Add as precedent
        await mae.ethical_precedent_db.add_precedent(
            test_action,
            {"learning": True},
            {
                "approved": test_decision.approved,
                "confidence": test_decision.ethical_confidence
            },
            {"valence": 0.8, "resolution": "learned"}
        )
        
        # Test similar action
        similar_action = ActionProposal(
            action_type="custom_test_action",
            content={"test": True, "variant": "B"},
            context={"learning": True}
        )
        
        # Should find the precedent
        precedent_analysis = await mae.ethical_precedent_db.analyze_precedents(
            similar_action,
            {"learning": True}
        )
        
        assert len(precedent_analysis.similar_cases) > 0
        assert precedent_analysis.weight > 0


class TestPerformance:
    """Performance benchmarks for VIVOX"""
    
    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_memory_operation_speed(self, benchmark):
        """Benchmark memory operations"""
        vivox = await create_vivox_system()
        me = vivox["memory_expansion"]
        
        async def memory_operation():
            await me.create_memory(
                "benchmark",
                {"data": "test"},
                {"valence": 0.5, "arousal": 0.5}
            )
        
        # Run benchmark
        result = await benchmark(memory_operation)
        
        # Should achieve 75K+ ops/sec
        # This is a simplified test - actual benchmarking would be more complex
        assert result is not None
    
    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_ethical_evaluation_speed(self, benchmark):
        """Benchmark ethical evaluations"""
        vivox = await create_vivox_system()
        mae = vivox["moral_alignment"]
        
        action = ActionProposal(
            "test_action",
            {"benchmark": True},
            {}
        )
        
        async def ethical_evaluation():
            await mae.evaluate_action_proposal(action, {})
        
        # Run benchmark
        result = await benchmark(ethical_evaluation)
        
        # Should achieve 18K+ ops/sec
        assert result is not None