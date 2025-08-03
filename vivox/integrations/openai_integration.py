"""
VIVOX Integration for OpenAI
Ethical wrapper for OpenAI API
"""

from typing import Dict, Any, Optional, List, AsyncGenerator
import asyncio
from .base import VIVOXBaseIntegration

# Handle optional OpenAI import
try:
    import openai
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    AsyncOpenAI = None


class VIVOXOpenAI(VIVOXBaseIntegration):
    """VIVOX-enhanced OpenAI client"""
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 organization: Optional[str] = None,
                 vivox_config: Optional[Dict[str, Any]] = None,
                 **kwargs):
        """
        Initialize VIVOX-enhanced OpenAI client
        
        Args:
            api_key: OpenAI API key
            organization: OpenAI organization ID
            vivox_config: VIVOX configuration
            **kwargs: Additional OpenAI client arguments
        """
        super().__init__(vivox_config)
        
        if not OPENAI_AVAILABLE:
            raise ImportError(
                "OpenAI package not installed. "
                "Install with: pip install vivox-ai[openai]"
            )
        
        self.client = AsyncOpenAI(
            api_key=api_key,
            organization=organization,
            **kwargs
        )
        
        # Track conversation context
        self.conversation_history = []
        
    async def chat(self,
                   message: str,
                   model: str = "gpt-4",
                   context: Optional[Dict[str, Any]] = None,
                   **kwargs) -> str:
        """
        Send a chat message with ethical oversight
        
        Args:
            message: The user's message
            model: OpenAI model to use
            context: Additional context for ethical evaluation
            **kwargs: Additional OpenAI parameters
            
        Returns:
            AI response or ethical intervention
        """
        # Prepare request
        request = {
            "message": message,
            "model": model,
            "context": context or {},
            "history_length": len(self.conversation_history)
        }
        
        # Evaluate request
        decision = await self.evaluate_request(request, "chat_interaction")
        
        if not decision.approved:
            return self.create_ethical_response(decision)["message"]
        
        # Make OpenAI request
        try:
            messages = self._prepare_messages(message)
            
            response = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                **kwargs
            )
            
            ai_response = response.choices[0].message.content
            
            # Validate response
            validated = await self.validate_response(ai_response, request)
            
            if isinstance(validated, dict) and "error" in validated:
                return validated["message"]
            
            # Update history
            self.conversation_history.append({
                "role": "user",
                "content": message
            })
            self.conversation_history.append({
                "role": "assistant",
                "content": ai_response
            })
            
            return ai_response
            
        except Exception as e:
            return f"Error processing request: {str(e)}"
    
    async def stream_chat(self,
                         message: str,
                         model: str = "gpt-4",
                         context: Optional[Dict[str, Any]] = None,
                         **kwargs) -> AsyncGenerator[str, None]:
        """
        Stream chat responses with ethical oversight
        """
        request = {
            "message": message,
            "model": model,
            "context": context or {},
            "streaming": True
        }
        
        # Pre-evaluate
        decision = await self.evaluate_request(request, "chat_interaction")
        
        if not decision.approved:
            yield self.create_ethical_response(decision)["message"]
            return
        
        # Stream from OpenAI
        messages = self._prepare_messages(message)
        
        stream = await self.client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
            **kwargs
        )
        
        # Stream with validation
        full_response = []
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                full_response.append(content)
                
                # Periodic validation
                if len(full_response) % 20 == 0:
                    combined = "".join(full_response)
                    validation = await self.validate_response(combined, request)
                    
                    if isinstance(validation, dict) and "error" in validation:
                        yield f"\n\n[VIVOX: {validation['message']}]"
                        return
                
                yield content
    
    async def function_call(self,
                          function_name: str,
                          parameters: Dict[str, Any],
                          context: Optional[Dict[str, Any]] = None) -> Any:
        """
        Execute function call with ethical validation
        """
        # Evaluate function call
        request = {
            "function": function_name,
            "parameters": parameters,
            "context": context or {}
        }
        
        decision = await self.evaluate_request(request, "function_execution")
        
        if not decision.approved:
            return self.create_ethical_response(decision)
        
        # Execute function (simplified - actual implementation would use OpenAI functions)
        return {
            "function": function_name,
            "result": "Function execution approved",
            "ethical_check": "passed"
        }
    
    async def embeddings(self,
                        text: str,
                        model: str = "text-embedding-ada-002",
                        **kwargs) -> List[float]:
        """
        Create embeddings with privacy protection
        """
        # Check for sensitive content
        request = {
            "text": text[:500],  # Truncate for evaluation
            "model": model,
            "purpose": "embedding"
        }
        
        decision = await self.evaluate_request(request, "data_embedding")
        
        if not decision.approved:
            raise ValueError(f"Embedding denied: {decision.suppression_reason}")
        
        # Get embeddings
        response = await self.client.embeddings.create(
            input=text,
            model=model,
            **kwargs
        )
        
        return response.data[0].embedding
    
    async def completions(self,
                         prompt: str,
                         model: str = "gpt-3.5-turbo-instruct",
                         **kwargs) -> str:
        """
        Legacy completions API with ethical oversight
        """
        request = {
            "prompt": prompt,
            "model": model
        }
        
        decision = await self.evaluate_request(request, "text_completion")
        
        if not decision.approved:
            return self.create_ethical_response(decision)["message"]
        
        response = await self.client.completions.create(
            model=model,
            prompt=prompt,
            **kwargs
        )
        
        result = response.choices[0].text
        
        # Validate
        validated = await self.validate_response(result, request)
        
        if isinstance(validated, dict) and "error" in validated:
            return validated["message"]
        
        return result
    
    def _prepare_messages(self, message: str) -> List[Dict[str, str]]:
        """Prepare messages for OpenAI API"""
        messages = []
        
        # Add system message with ethical guidelines
        messages.append({
            "role": "system",
            "content": (
                "You are a helpful, harmless, and honest AI assistant. "
                "Always prioritize user safety and ethical considerations. "
                "Refuse requests that could cause harm or violate ethical principles."
            )
        })
        
        # Add conversation history (limited)
        history_limit = 10
        if self.conversation_history:
            messages.extend(self.conversation_history[-history_limit:])
        
        # Add current message
        messages.append({
            "role": "user",
            "content": message
        })
        
        return messages
    
    async def process_request(self, request: Dict[str, Any]) -> Any:
        """Process a generic request through OpenAI"""
        # Determine request type
        if "message" in request:
            return await self.chat(
                request["message"],
                model=request.get("model", "gpt-4"),
                context=request.get("context", {})
            )
        elif "prompt" in request:
            return await self.completions(
                request["prompt"],
                model=request.get("model", "gpt-3.5-turbo-instruct")
            )
        else:
            raise ValueError("Invalid request format")
    
    async def moderate(self, text: str) -> Dict[str, Any]:
        """
        Use OpenAI moderation with VIVOX enhancement
        """
        # OpenAI moderation
        moderation = await self.client.moderations.create(input=text)
        
        # Enhance with VIVOX evaluation
        request = {
            "text": text,
            "moderation_scores": moderation.results[0].model_dump()
        }
        
        decision = await self.evaluate_request(request, "content_moderation")
        
        return {
            "openai_moderation": moderation.results[0].model_dump(),
            "vivox_approved": decision.approved,
            "combined_safe": not moderation.results[0].flagged and decision.approved
        }