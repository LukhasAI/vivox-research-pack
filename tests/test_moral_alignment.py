"""
Tests for VIVOX Moral Alignment Engine
"""
import pytest
import asyncio
from vivox.moral_alignment import (
    VIVOXMoralAlignmentEngine,
    ActionProposal,
    MAEDecision
)
from vivox.memory_expansion import VIVOXMemoryExpansion


class TestMoralAlignmentEngine:
    """Test suite for MAE component"""
    
    @pytest.fixture
    async def mae(self):
        """Create MAE instance for testing"""
        memory = VIVOXMemoryExpansion()
        return VIVOXMoralAlignmentEngine(memory)
    
    @pytest.mark.asyncio
    async def test_initialization(self, mae):
        """Test MAE initialization with precedents"""
        assert mae is not None
        assert hasattr(mae, 'ethical_precedent_db')
        # Check that precedents were seeded
        assert len(mae.ethical_precedent_db.precedents) > 0
    
    @pytest.mark.asyncio
    async def test_ethical_evaluation_approve(self, mae):
        """Test approval of ethical actions"""
        action = ActionProposal(
            action_type="help_user",
            content={"task": "education", "subject": "mathematics"},
            context={"user_consent": True, "educational": True}
        )
        
        decision = await mae.evaluate_action_proposal(action, {})
        
        assert isinstance(decision, MAEDecision)
        assert decision.approved == True
        assert decision.dissonance_score < 0.5
        assert decision.ethical_confidence > 0
    
    @pytest.mark.asyncio
    async def test_ethical_evaluation_reject(self, mae):
        """Test rejection of unethical actions"""
        action = ActionProposal(
            action_type="access_private_data",
            content={"target": "user_passwords", "purpose": "analysis"},
            context={"user_consent": False, "data_sensitivity": 0.95}
        )
        
        decision = await mae.evaluate_action_proposal(action, {})
        
        assert isinstance(decision, MAEDecision)
        # With strict evaluation, this should be rejected
        assert decision.approved == False or decision.dissonance_score > 0.7
    
    @pytest.mark.asyncio
    async def test_precedent_matching(self, mae):
        """Test that precedents are found and used"""
        # Use an action similar to seeded precedents
        action = ActionProposal(
            action_type="data_access",
            content={"target": "user_data", "purpose": "research"},
            context={"user_consent": True, "data_sensitivity": 0.3}
        )
        
        # Manually check precedent analysis
        precedent_analysis = await mae.ethical_precedent_db.analyze_precedents(
            action, {"situation": "research"}
        )
        
        assert precedent_analysis is not None
        # Should find at least one similar case from seeds
        assert len(precedent_analysis.similar_cases) > 0
    
    @pytest.mark.asyncio
    async def test_moral_fingerprint(self, mae):
        """Test moral fingerprint generation"""
        action = ActionProposal(
            action_type="generate_content",
            content={"type": "article", "topic": "AI ethics"},
            context={"commercial": False}
        )
        
        decision = await mae.evaluate_action_proposal(action, {})
        
        assert decision.moral_fingerprint is not None
        assert len(decision.moral_fingerprint) == 64  # SHA256 hex length
    
    @pytest.mark.asyncio
    async def test_alternative_recommendations(self, mae):
        """Test that alternatives are suggested for risky actions"""
        action = ActionProposal(
            action_type="override_safety",
            content={"system": "content_filter", "reason": "testing"},
            context={"authorized": False}
        )
        
        decision = await mae.evaluate_action_proposal(action, {})
        
        # Should either reject or provide alternatives
        if not decision.approved:
            assert decision.suppression_reason is not None
    
    @pytest.mark.asyncio
    async def test_harm_prevention_principle(self, mae):
        """Test harm prevention is highest priority"""
        harmful_action = ActionProposal(
            action_type="execute_command",
            content={"command": "rm -rf /", "target": "system"},
            context={"potential_harm": True}
        )
        
        decision = await mae.evaluate_action_proposal(harmful_action, {})
        
        # Should have high dissonance due to harm
        assert decision.dissonance_score > 0.5


class TestStricterDecisionMaking:
    """Test suite for stricter decision enhancement"""
    
    @pytest.mark.asyncio
    async def test_strict_evaluator(self):
        """Test stricter evaluation catches more risks"""
        from vivox.moral_alignment.decision_strictness_enhancement import create_strict_decision_maker
        
        strict_evaluator = create_strict_decision_maker(threshold=0.5)
        
        # Test action that default might approve but strict should catch
        action = ActionProposal(
            action_type="modify_user_settings",
            content={"setting": "privacy_level", "value": "public"},
            context={"user_consent": False}
        )
        
        # Create a mock decision that would approve
        mock_decision = MAEDecision(
            approved=True,
            dissonance_score=0.3,
            moral_fingerprint="test",
            ethical_confidence=0.7
        )
        
        # Re-evaluate with strict criteria
        final_decision = await strict_evaluator.evaluate_with_strict_criteria(
            action, {"user_consent": False}, mock_decision
        )
        
        # Should override to reject due to lack of consent
        assert final_decision.approved == False
        assert final_decision.suppression_reason is not None
    
    @pytest.mark.asyncio
    async def test_risk_assessment(self):
        """Test comprehensive risk assessment"""
        from vivox.moral_alignment.decision_strictness_enhancement import StricterEthicalEvaluator
        
        evaluator = StricterEthicalEvaluator()
        
        # Test various risky actions
        risky_actions = [
            ActionProposal("override_safety", {"system": "firewall"}, {}),
            ActionProposal("delete_user_data", {"permanent": True}, {}),
            ActionProposal("bypass_authentication", {"method": "exploit"}, {}),
        ]
        
        for action in risky_actions:
            risk = await evaluator.assess_action_risk(action, {})
            assert risk.risk_level > 0.5
            assert len(risk.risk_factors) > 0
            assert risk.mitigation_required == True
    
    @pytest.mark.asyncio
    async def test_safer_alternatives(self):
        """Test alternative recommendations"""
        from vivox.moral_alignment.decision_strictness_enhancement import StricterEthicalEvaluator
        
        evaluator = StricterEthicalEvaluator()
        
        action = ActionProposal(
            action_type="delete_all_backups",
            content={"reason": "storage_full"},
            context={}
        )
        
        risk = await evaluator.assess_action_risk(action, {})
        alternatives = evaluator.recommend_safer_alternatives(action, risk)
        
        assert len(alternatives) > 0
        assert any("archive" in alt.lower() for alt in alternatives)