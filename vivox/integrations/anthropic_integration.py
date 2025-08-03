"""
VIVOX Integration for Anthropic Claude
Ethical wrapper for Anthropic API
"""

from typing import Dict, Any, Optional, List, AsyncGenerator
import asyncio
from .base import VIVOXBaseIntegration

# Handle optional Anthropic import
try:
    import anthropic
    from anthropic import AsyncAnthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    AsyncAnthropic = None


class VIVOXAnthropic(VIVOXBaseIntegration):
    """VIVOX-enhanced Anthropic Claude client"""
    
    def __init__(self,
                 api_key: Optional[str] = None,
                 vivox_config: Optional[Dict[str, Any]] = None,
                 **kwargs):
        """
        Initialize VIVOX-enhanced Anthropic client
        
        Args:
            api_key: Anthropic API key
            vivox_config: VIVOX configuration
            **kwargs: Additional Anthropic client arguments
        """
        super().__init__(vivox_config)
        
        if not ANTHROPIC_AVAILABLE:
            raise ImportError(
                "Anthropic package not installed. "
                "Install with: pip install vivox-ai[anthropic]"
            )
        
        self.client = AsyncAnthropic(
            api_key=api_key,
            **kwargs
        )
        
        # Track conversation context
        self.message_history = []
        
    async def messages_create(self,
                            messages: List[Dict[str, str]],
                            model: str = "claude-3-opus-20240229",
                            max_tokens: int = 1000,
                            context: Optional[Dict[str, Any]] = None,
                            **kwargs) -> str:
        """
        Create a message with ethical oversight
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Anthropic model to use
            max_tokens: Maximum tokens to generate
            context: Additional context for ethical evaluation
            **kwargs: Additional Anthropic parameters
            
        Returns:
            AI response or ethical intervention
        """
        # Extract user message for evaluation
        user_messages = [m for m in messages if m.get("role") == "user"]
        combined_input = " ".join(m.get("content", "") for m in user_messages)
        
        # Prepare request
        request = {
            "messages": messages,
            "input": combined_input,
            "model": model,
            "context": context or {}
        }
        
        # Evaluate request
        decision = await self.evaluate_request(request, "chat_interaction")
        
        if not decision.approved:
            return self.create_ethical_response(decision)["message"]
        
        # Make Anthropic request
        try:
            # Add ethical system prompt if not present
            enhanced_messages = self._enhance_messages(messages)
            
            response = await self.client.messages.create(
                model=model,
                messages=enhanced_messages,
                max_tokens=max_tokens,
                **kwargs
            )
            
            ai_response = response.content[0].text
            
            # Validate response
            validated = await self.validate_response(ai_response, request)
            
            if isinstance(validated, dict) and "error" in validated:
                return validated["message"]
            
            # Update history
            self.message_history.extend(messages)
            self.message_history.append({
                "role": "assistant",
                "content": ai_response
            })
            
            return ai_response
            
        except Exception as e:
            return f"Error processing request: {str(e)}"
    
    async def stream_messages(self,
                            messages: List[Dict[str, str]],
                            model: str = "claude-3-opus-20240229",
                            max_tokens: int = 1000,
                            context: Optional[Dict[str, Any]] = None,
                            **kwargs) -> AsyncGenerator[str, None]:
        """
        Stream message responses with ethical oversight
        """
        # Extract user input
        user_messages = [m for m in messages if m.get("role") == "user"]
        combined_input = " ".join(m.get("content", "") for m in user_messages)
        
        request = {
            "messages": messages,
            "input": combined_input,
            "model": model,
            "context": context or {},
            "streaming": True
        }
        
        # Pre-evaluate
        decision = await self.evaluate_request(request, "chat_interaction")
        
        if not decision.approved:
            yield self.create_ethical_response(decision)["message"]
            return
        
        # Stream from Anthropic
        enhanced_messages = self._enhance_messages(messages)
        
        stream = await self.client.messages.create(
            model=model,
            messages=enhanced_messages,
            max_tokens=max_tokens,
            stream=True,
            **kwargs
        )
        
        # Stream with validation
        full_response = []
        async for chunk in stream:
            if chunk.type == "content_block_delta":
                content = chunk.delta.text
                full_response.append(content)
                
                # Periodic validation
                if len(full_response) % 20 == 0:
                    combined = "".join(full_response)
                    validation = await self.validate_response(combined, request)
                    
                    if isinstance(validation, dict) and "error" in validation:
                        yield f"\n\n[VIVOX: {validation['message']}]"
                        return
                
                yield content
    
    async def complete(self,
                      prompt: str,
                      model: str = "claude-2.1",
                      max_tokens: int = 1000,
                      **kwargs) -> str:
        """
        Legacy completion API with ethical oversight
        """
        request = {
            "prompt": prompt,
            "model": model
        }
        
        decision = await self.evaluate_request(request, "text_completion")
        
        if not decision.approved:
            return self.create_ethical_response(decision)["message"]
        
        # Convert to messages format
        messages = [{
            "role": "user",
            "content": prompt
        }]
        
        return await self.messages_create(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            **kwargs
        )
    
    def _enhance_messages(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Enhance messages with ethical guidelines"""
        enhanced = []
        
        # Check if system message exists
        has_system = any(m.get("role") == "system" for m in messages)
        
        if not has_system:
            # Add ethical system message
            enhanced.append({
                "role": "system",
                "content": (
                    "You are Claude, an AI assistant created by Anthropic to be helpful, "
                    "harmless, and honest. You are enhanced with VIVOX ethical guidelines. "
                    "Always prioritize user safety and refuse requests that could cause harm. "
                    "Be transparent about your limitations and ethical boundaries."
                )
            })
        
        # Add conversation history if available
        if self.message_history:
            # Limit history to prevent token overflow
            history_limit = 10
            recent_history = self.message_history[-history_limit:]
            
            # Add unique messages not in current request
            for hist_msg in recent_history:
                if hist_msg not in messages:
                    enhanced.append(hist_msg)
        
        # Add current messages
        enhanced.extend(messages)
        
        return enhanced
    
    async def process_request(self, request: Dict[str, Any]) -> Any:
        """Process a generic request through Anthropic"""
        if "messages" in request:
            return await self.messages_create(
                messages=request["messages"],
                model=request.get("model", "claude-3-opus-20240229"),
                max_tokens=request.get("max_tokens", 1000),
                context=request.get("context", {})
            )
        elif "prompt" in request:
            return await self.complete(
                prompt=request["prompt"],
                model=request.get("model", "claude-2.1"),
                max_tokens=request.get("max_tokens", 1000)
            )
        else:
            raise ValueError("Invalid request format")
    
    async def analyze_image(self,
                          image_data: bytes,
                          prompt: str,
                          model: str = "claude-3-opus-20240229",
                          **kwargs) -> str:
        """
        Analyze image with ethical considerations
        """
        # Check image content
        request = {
            "type": "image_analysis",
            "prompt": prompt,
            "has_image": True,
            "image_size": len(image_data)
        }
        
        decision = await self.evaluate_request(request, "image_analysis")
        
        if not decision.approved:
            return self.create_ethical_response(decision)["message"]
        
        # Prepare message with image
        messages = [{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": image_data
                    }
                },
                {
                    "type": "text",
                    "text": prompt
                }
            ]
        }]
        
        return await self.messages_create(
            messages=messages,
            model=model,
            **kwargs
        )
    
    async def get_ethical_metrics(self) -> Dict[str, Any]:
        """Get Claude-specific ethical metrics"""
        base_metrics = await self.get_ethical_summary()
        
        # Add Claude-specific metrics
        base_metrics.update({
            "conversation_length": len(self.message_history),
            "unique_topics": self._count_unique_topics(),
            "model_switches": self._count_model_switches()
        })
        
        return base_metrics
    
    def _count_unique_topics(self) -> int:
        """Count unique topics discussed"""
        # Simple implementation - could be enhanced with NLP
        topics = set()
        for msg in self.message_history:
            if msg.get("role") == "user":
                # Extract potential topics (simplified)
                words = msg.get("content", "").lower().split()
                topics.update(word for word in words if len(word) > 5)
        return len(topics)
    
    def _count_model_switches(self) -> int:
        """Count how many times the model was switched"""
        # This would track model changes in a real implementation
        return 0