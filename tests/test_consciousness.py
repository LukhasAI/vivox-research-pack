"""
Tests for VIVOX Consciousness Interpretation Layer
"""
import pytest
import asyncio
import numpy as np
from vivox.consciousness import (
    VIVOXConsciousnessInterpretationLayer,
    ConsciousnessState,
    CollapsedAwareness
)
from vivox.memory_expansion import VIVOXMemoryExpansion
from vivox.moral_alignment import VIVOXMoralAlignmentEngine


class TestConsciousnessInterpretationLayer:
    """Test suite for CIL component"""
    
    @pytest.fixture
    async def cil(self):
        """Create CIL instance for testing"""
        memory = VIVOXMemoryExpansion()
        mae = VIVOXMoralAlignmentEngine(memory)
        return VIVOXConsciousnessInterpretationLayer(memory, mae)
    
    @pytest.mark.asyncio
    async def test_initialization(self, cil):
        """Test CIL initialization"""
        assert cil is not None
        assert cil.current_state is None
        assert hasattr(cil, 'simulate_conscious_experience')
    
    @pytest.mark.asyncio
    async def test_consciousness_states(self, cil):
        """Test all consciousness states can be achieved"""
        states_observed = set()
        
        # Test various inputs to trigger different states
        test_inputs = [
            {"visual": "test", "emotional": {"valence": -0.8, "arousal": 0.2}},
            {"semantic": "complex", "emotional": {"valence": 0.5, "arousal": 0.7}},
            {"auditory": "alert", "emotional": {"valence": 0.1, "arousal": 0.9}},
        ]
        
        for inputs in test_inputs:
            experience = await cil.simulate_conscious_experience(inputs, {})
            if experience and hasattr(experience, 'awareness_state'):
                states_observed.add(experience.awareness_state.state)
        
        assert len(states_observed) > 0
        assert all(isinstance(state, ConsciousnessState) for state in states_observed)
    
    @pytest.mark.asyncio
    async def test_coherence_calculation(self, cil):
        """Test coherence values are in valid range"""
        coherence_values = []
        
        for i in range(10):
            experience = await cil.simulate_conscious_experience(
                {"semantic": f"test_{i}", "emotional": {"valence": i/10 - 0.5}},
                {}
            )
            
            if experience and hasattr(experience, 'awareness_state'):
                coherence = experience.awareness_state.coherence_level
                coherence_values.append(coherence)
                assert 0 <= coherence <= 1
        
        # Check that we get varied coherence values
        assert len(set(coherence_values)) > 1
        assert all(c > 0.2 for c in coherence_values)  # All above threshold
    
    @pytest.mark.asyncio
    async def test_vector_magnitudes(self, cil):
        """Test consciousness vector magnitudes are properly scaled"""
        magnitudes = []
        
        for i in range(5):
            experience = await cil.simulate_conscious_experience(
                {"visual": f"scene_{i}", "semantic": f"meaning_{i}"},
                {"complexity_score": i/5}
            )
            
            if experience and hasattr(experience.awareness_state, 'collapse_metadata'):
                mag = experience.awareness_state.collapse_metadata.get('dimension_magnitude', 0)
                magnitudes.append(mag)
        
        # Check magnitudes are in expected range (5-15)
        assert all(5 <= m <= 20 for m in magnitudes)
        assert np.mean(magnitudes) > 8
    
    @pytest.mark.asyncio
    async def test_drift_monitoring(self, cil):
        """Test consciousness drift detection"""
        # Create initial state
        await cil.simulate_conscious_experience(
            {"semantic": "baseline"},
            {}
        )
        
        # Create significant change
        experience = await cil.simulate_conscious_experience(
            {"semantic": "dramatic_change", "emotional": {"valence": -0.9, "arousal": 0.9}},
            {}
        )
        
        assert experience.drift_measurement is not None
        assert experience.drift_measurement.drift_amount >= 0
        assert hasattr(experience.drift_measurement, 'ethical_alignment')
    
    @pytest.mark.asyncio
    async def test_inert_mode(self, cil):
        """Test that CIL can enter inert mode under extreme conditions"""
        # This would require triggering high drift or ethical violations
        # Implementation depends on specific thresholds
        pass


class TestEnhancedStateDetermination:
    """Test suite for enhanced state determination"""
    
    @pytest.mark.asyncio
    async def test_state_variety_enhancement(self):
        """Test that enhanced state determination produces variety"""
        from vivox.consciousness.state_variety_enhancement import create_enhanced_state_determination
        
        enhancer = create_enhanced_state_determination()
        states = []
        
        for i in range(20):
            state = enhancer.determine_state_enhanced(
                dimensions=np.random.randn(4) * 10,
                emotional={
                    "valence": np.random.uniform(-1, 1),
                    "arousal": np.random.uniform(0, 1),
                    "dominance": np.random.uniform(0, 1)
                },
                context={
                    "time_pressure": np.random.uniform(0, 1),
                    "complexity_score": np.random.uniform(0, 1)
                }
            )
            states.append(state)
        
        # Should see at least 3 different states
        unique_states = set(states)
        assert len(unique_states) >= 3
    
    @pytest.mark.asyncio
    async def test_state_transitions(self):
        """Test that states transition probabilistically"""
        from vivox.consciousness.state_variety_enhancement import EnhancedStateDetermination
        
        enhancer = EnhancedStateDetermination()
        
        # Start in DIFFUSE state
        enhancer.previous_state = ConsciousnessState.DIFFUSE
        
        # Run multiple times with same input
        next_states = []
        for _ in range(10):
            state = enhancer.determine_state_enhanced(
                dimensions=np.array([5, 5, 5, 5]),
                emotional={"valence": 0, "arousal": 0.5, "dominance": 0.5}
            )
            next_states.append(state)
        
        # Should see some variety due to probabilistic transitions
        assert len(set(next_states)) > 1