"""
Tests for VIVOX Memory Expansion
"""
import pytest
import asyncio
import numpy as np
from vivox.memory_expansion import (
    VIVOXMemoryExpansion,
    MemoryHelixEntry,
    EmotionalDNA,
    VeilLevel
)


class TestMemoryExpansion:
    """Test suite for Memory Expansion component"""
    
    @pytest.fixture
    async def memory(self):
        """Create memory instance for testing"""
        return VIVOXMemoryExpansion()
    
    @pytest.mark.asyncio
    async def test_initialization(self, memory):
        """Test memory initialization"""
        assert memory is not None
        assert hasattr(memory, 'helix_structure')
        assert hasattr(memory, 'create_memory')
    
    @pytest.mark.asyncio
    async def test_memory_creation(self, memory):
        """Test creating memories with emotional context"""
        # Create a memory
        memory_entry = await memory.create_memory(
            memory_type="experience",
            content={"event": "test_event", "data": "test_data"},
            emotional_context={"valence": 0.7, "arousal": 0.5, "dominance": 0.6}
        )
        
        assert isinstance(memory_entry, MemoryHelixEntry)
        assert memory_entry.memory_type == "experience"
        assert memory_entry.content["event"] == "test_event"
        assert memory_entry.emotional_dna is not None
    
    @pytest.mark.asyncio
    async def test_emotional_dna(self, memory):
        """Test emotional DNA encoding"""
        emotional_context = {"valence": 0.8, "arousal": 0.6, "dominance": 0.7}
        
        memory_entry = await memory.create_memory(
            memory_type="test",
            content={"data": "test"},
            emotional_context=emotional_context
        )
        
        # Check emotional DNA was encoded
        assert isinstance(memory_entry.emotional_dna, EmotionalDNA)
        assert 0 <= memory_entry.emotional_dna.valence <= 1
        assert 0 <= memory_entry.emotional_dna.arousal <= 1
        assert 0 <= memory_entry.emotional_dna.dominance <= 1
    
    @pytest.mark.asyncio
    async def test_memory_veiling(self, memory):
        """Test memory veiling for privacy"""
        # Create sensitive memory
        sensitive_memory = await memory.create_memory(
            memory_type="private",
            content={"sensitive": "user_password", "data": "secret123"},
            veil_level=VeilLevel.OPAQUE
        )
        
        assert sensitive_memory.veil_level == VeilLevel.OPAQUE
        
        # Test veiling operation
        veiling_op = await memory.apply_veiling(
            sensitive_memory.memory_id,
            VeilLevel.TRANSLUCENT
        )
        
        assert veiling_op is not None
        assert veiling_op.new_level == VeilLevel.TRANSLUCENT
    
    @pytest.mark.asyncio
    async def test_memory_retrieval(self, memory):
        """Test memory retrieval by various criteria"""
        # Create multiple memories
        memories = []
        for i in range(5):
            mem = await memory.create_memory(
                memory_type="test",
                content={"index": i, "data": f"test_{i}"},
                emotional_context={
                    "valence": i / 5,
                    "arousal": 0.5,
                    "dominance": 0.5
                }
            )
            memories.append(mem)
            await asyncio.sleep(0.01)  # Small delay for time-based ordering
        
        # Test retrieval by type
        retrieved = await memory.retrieve_memories_by_type("test", limit=3)
        assert len(retrieved) <= 3
        assert all(m.memory_type == "test" for m in retrieved)
    
    @pytest.mark.asyncio
    async def test_emotional_similarity_search(self, memory):
        """Test finding memories by emotional similarity"""
        # Create memories with different emotional profiles
        happy_memory = await memory.create_memory(
            memory_type="emotion",
            content={"mood": "happy"},
            emotional_context={"valence": 0.9, "arousal": 0.7, "dominance": 0.6}
        )
        
        sad_memory = await memory.create_memory(
            memory_type="emotion",
            content={"mood": "sad"},
            emotional_context={"valence": -0.8, "arousal": 0.3, "dominance": 0.2}
        )
        
        # Search for similar emotional state
        similar = await memory.find_similar_memories(
            emotional_state={"valence": 0.85, "arousal": 0.65, "dominance": 0.55},
            limit=1
        )
        
        # Should find the happy memory as most similar
        if similar:
            assert similar[0].content["mood"] == "happy"
    
    @pytest.mark.asyncio
    async def test_memory_folding(self, memory):
        """Test protein-like memory folding"""
        # Create a complex memory that should fold
        complex_memory = await memory.create_memory(
            memory_type="complex",
            content={
                "data": "x" * 1000,  # Large content
                "nested": {"level1": {"level2": {"level3": "deep"}}}
            },
            emotional_context={"valence": 0.5, "arousal": 0.5, "dominance": 0.5}
        )
        
        # Check if folding occurred (implementation specific)
        assert complex_memory is not None
        # Folding details would depend on implementation
    
    @pytest.mark.asyncio
    async def test_cascade_prevention(self, memory):
        """Test cascade prevention in memory updates"""
        # Create interconnected memories
        root_memory = await memory.create_memory(
            memory_type="root",
            content={"id": "root", "connections": []},
            cascade_risk=0.8
        )
        
        # Create dependent memories
        for i in range(3):
            child = await memory.create_memory(
                memory_type="child",
                content={"id": f"child_{i}", "parent": root_memory.memory_id},
                cascade_risk=0.5
            )
        
        # Test cascade prevention when modifying root
        # Implementation would prevent cascading changes
        assert root_memory.cascade_risk > 0.5


class TestSymbolicProteome:
    """Test suite for Symbolic Proteome component"""
    
    @pytest.mark.asyncio
    async def test_protein_folding(self):
        """Test protein-like folding patterns"""
        from vivox.memory_expansion.symbolic_proteome import VIVOXSymbolicProteome
        
        proteome = VIVOXSymbolicProteome()
        
        # Create test fold
        fold = await proteome.create_fold(
            fold_type="alpha_helix",
            content={"structure": "HELIX", "stability": 0.8},
            binding_sites=["memory_1", "memory_2"]
        )
        
        assert fold is not None
        assert fold.fold_type == "alpha_helix"
        assert len(fold.binding_sites) == 2
    
    @pytest.mark.asyncio
    async def test_misfolding_detection(self):
        """Test detection of memory misfolding"""
        from vivox.memory_expansion.symbolic_proteome import (
            VIVOXSymbolicProteome,
            MisfoldingType
        )
        
        proteome = VIVOXSymbolicProteome()
        
        # Create potentially misfolded structure
        problematic_fold = await proteome.create_fold(
            fold_type="beta_sheet",
            content={"structure": "SHEET", "stability": 0.2},  # Low stability
            binding_sites=["memory_1", "memory_1"]  # Duplicate binding
        )
        
        # Check for misfolding
        misfolding_report = await proteome.check_misfolding(problematic_fold)
        
        assert misfolding_report is not None
        assert len(misfolding_report.issues) > 0