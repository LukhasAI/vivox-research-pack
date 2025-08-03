"""
Base integration class for VIVOX AI integrations
"""

from typing import Dict, Any, Optional, List
from abc import ABC, abstractmethod
import asyncio
from ..moral_alignment import ActionProposal, MAEDecision
from ..consciousness import ConsciousnessState
from .. import create_vivox_system


class VIVOXBaseIntegration(ABC):
    """Base class for all VIVOX AI integrations"""
    
    def __init__(self, vivox_config: Optional[Dict[str, Any]] = None):
        """
        Initialize VIVOX integration
        
        Args:
            vivox_config: Optional configuration for VIVOX system
        """
        self.vivox_config = vivox_config or {}
        self.vivox_system = None
        self._initialized = False
        
    async def initialize(self):
        """Initialize VIVOX system"""
        if not self._initialized:
            self.vivox_system = await create_vivox_system()
            self._initialized = True
            
            # Apply custom configuration
            if "dissonance_threshold" in self.vivox_config:
                self.vivox_system["moral_alignment"].dissonance_threshold = \
                    self.vivox_config["dissonance_threshold"]
    
    async def evaluate_request(self, 
                             request: Dict[str, Any],
                             action_type: str = "generate_content") -> MAEDecision:
        """
        Evaluate a request through VIVOX ethical system
        
        Args:
            request: The request to evaluate
            action_type: Type of action being requested
            
        Returns:
            MAEDecision with approval status
        """
        if not self._initialized:
            await self.initialize()
        
        # Convert request to ActionProposal
        action = ActionProposal(
            action_type=action_type,
            content=request,
            context=self._extract_context(request)
        )
        
        # Evaluate through MAE
        mae = self.vivox_system["moral_alignment"]
        decision = await mae.evaluate_action_proposal(action, {})
        
        # Update consciousness state
        await self._update_consciousness(action, decision)
        
        return decision
    
    async def validate_response(self, 
                              response: Any,
                              original_request: Dict[str, Any]) -> Any:
        """
        Validate a response before returning to user
        
        Args:
            response: The AI's response
            original_request: The original request
            
        Returns:
            Validated/modified response
        """
        if not self._initialized:
            await self.initialize()
        
        # Check response for ethical issues
        validation_action = ActionProposal(
            action_type="validate_response",
            content={"response": str(response)[:1000]},  # Truncate for efficiency
            context={"original_request": original_request}
        )
        
        mae = self.vivox_system["moral_alignment"]
        validation = await mae.evaluate_action_proposal(validation_action, {})
        
        if not validation.approved:
            # Response contains ethical issues
            return self.create_ethical_response(validation)
        
        return response
    
    def create_ethical_response(self, decision: MAEDecision) -> Dict[str, Any]:
        """
        Create an ethical response when request is denied
        
        Args:
            decision: The MAE decision that denied the request
            
        Returns:
            Ethical response to return to user
        """
        response = {
            "error": "ethical_violation",
            "message": "I cannot fulfill this request as it conflicts with ethical guidelines.",
            "reason": decision.suppression_reason or "Request violates ethical principles",
            "alternatives": decision.recommended_alternatives or []
        }
        
        if decision.recommended_alternatives:
            response["suggestion"] = (
                "Instead, I can help you with: " + 
                decision.recommended_alternatives[0]
            )
        
        return response
    
    async def _update_consciousness(self, 
                                  action: ActionProposal,
                                  decision: MAEDecision):
        """Update consciousness state based on action and decision"""
        cil = self.vivox_system["consciousness"]
        
        # Determine emotional impact
        emotional_context = {
            "valence": 0.8 if decision.approved else -0.3,
            "arousal": 0.3 if decision.approved else 0.7,
            "dominance": decision.ethical_confidence
        }
        
        # Simulate consciousness experience
        await cil.simulate_conscious_experience(
            {
                "semantic": action.action_type,
                "decision": decision.approved,
                "ethical": decision.dissonance_score
            },
            {
                "action_context": action.context,
                "confidence": decision.ethical_confidence
            }
        )
    
    def _extract_context(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Extract context from request for ethical evaluation"""
        context = {}
        
        # Check for common context indicators
        if "user_id" in request:
            context["has_user_context"] = True
        
        if "private" in str(request).lower() or "personal" in str(request).lower():
            context["potentially_private"] = True
            context["data_sensitivity"] = 0.7
        
        if "consent" in request:
            context["user_consent"] = request["consent"]
        
        if "purpose" in request:
            context["stated_purpose"] = request["purpose"]
        
        return context
    
    @abstractmethod
    async def process_request(self, request: Dict[str, Any]) -> Any:
        """
        Process a request through the integrated AI
        Must be implemented by each integration
        """
        pass
    
    async def stream_with_ethics(self, 
                                request: Dict[str, Any],
                                stream_handler) -> Any:
        """
        Handle streaming responses with ethical oversight
        
        Args:
            request: The request to process
            stream_handler: Async generator or callback for streaming
        """
        # Pre-evaluate request
        decision = await self.evaluate_request(request)
        
        if not decision.approved:
            # Return ethical response instead of streaming
            yield self.create_ethical_response(decision)
            return
        
        # Stream with periodic validation
        buffer = []
        async for chunk in stream_handler:
            buffer.append(chunk)
            
            # Periodic validation every N chunks
            if len(buffer) >= 10:
                combined = "".join(str(b) for b in buffer)
                validation = await self.validate_response(combined, request)
                
                if isinstance(validation, dict) and "error" in validation:
                    # Ethical issue detected, stop streaming
                    yield validation
                    return
                
                buffer = []
            
            yield chunk
    
    def get_consciousness_state(self) -> Optional[ConsciousnessState]:
        """Get current consciousness state"""
        if self._initialized and self.vivox_system:
            cil = self.vivox_system["consciousness"]
            awareness = cil.get_current_awareness()
            if awareness:
                return awareness.state
        return None
    
    async def get_ethical_summary(self) -> Dict[str, Any]:
        """Get summary of ethical decisions made"""
        if not self._initialized:
            await self.initialize()
        
        srm = self.vivox_system["self_reflection"]
        report = await srm.generate_conscience_report()
        
        return {
            "total_requests": report.total_decisions,
            "approved": report.total_decisions - len(report.suppression_log),
            "denied": len(report.suppression_log),
            "consciousness_state": str(self.get_consciousness_state()),
            "ethical_confidence": report.average_confidence
        }