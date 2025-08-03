"""
VIVOX Integration for Local Models
Ethical wrapper for locally-hosted AI models
"""

from typing import Dict, Any, Optional, List, Callable, AsyncGenerator
import asyncio
from abc import abstractmethod
from .base import VIVOXBaseIntegration


class VIVOXLocalModel(VIVOXBaseIntegration):
    """VIVOX wrapper for local AI models"""
    
    def __init__(self,
                 model: Any,
                 model_type: str = "generic",
                 vivox_config: Optional[Dict[str, Any]] = None,
                 **kwargs):
        """
        Initialize VIVOX wrapper for local model
        
        Args:
            model: The local model instance
            model_type: Type of model (generic, llama, gpt4all, etc.)
            vivox_config: VIVOX configuration
            **kwargs: Additional configuration
        """
        super().__init__(vivox_config)
        
        self.model = model
        self.model_type = model_type
        self.model_config = kwargs
        
        # Determine generation method based on model type
        self._setup_generation_method()
        
    def _setup_generation_method(self):
        """Setup the appropriate generation method based on model type"""
        if self.model_type == "llama":
            self.generate_method = self._generate_llama
        elif self.model_type == "gpt4all":
            self.generate_method = self._generate_gpt4all
        elif self.model_type == "transformers":
            self.generate_method = self._generate_transformers
        elif self.model_type == "ggml":
            self.generate_method = self._generate_ggml
        else:
            # Generic fallback
            self.generate_method = self._generate_generic
    
    async def generate(self,
                      prompt: str,
                      max_tokens: int = 1000,
                      temperature: float = 0.7,
                      context: Optional[Dict[str, Any]] = None,
                      **kwargs) -> str:
        """
        Generate text with ethical oversight
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            context: Additional context for ethical evaluation
            **kwargs: Model-specific parameters
            
        Returns:
            Generated text or ethical intervention
        """
        # Prepare request
        request = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "context": context or {},
            "model_type": self.model_type
        }
        
        # Evaluate request
        decision = await self.evaluate_request(request, "text_generation")
        
        if not decision.approved:
            return self.create_ethical_response(decision)["message"]
        
        try:
            # Add ethical prompt enhancement
            enhanced_prompt = self._enhance_prompt(prompt)
            
            # Generate using appropriate method
            response = await self.generate_method(
                enhanced_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )
            
            # Validate response
            validated = await self.validate_response(response, request)
            
            if isinstance(validated, dict) and "error" in validated:
                return validated["message"]
            
            return response
            
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    async def chat(self,
                   message: str,
                   history: Optional[List[Dict[str, str]]] = None,
                   context: Optional[Dict[str, Any]] = None,
                   **kwargs) -> str:
        """
        Chat interface with ethical oversight
        """
        # Build prompt from history
        prompt = self._build_chat_prompt(message, history)
        
        # Use generate with chat context
        chat_context = context or {}
        chat_context["chat_mode"] = True
        chat_context["history_length"] = len(history) if history else 0
        
        return await self.generate(
            prompt=prompt,
            context=chat_context,
            **kwargs
        )
    
    async def stream_generate(self,
                            prompt: str,
                            max_tokens: int = 1000,
                            context: Optional[Dict[str, Any]] = None,
                            **kwargs) -> AsyncGenerator[str, None]:
        """
        Stream generation with ethical oversight
        """
        # Pre-evaluate
        request = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "context": context or {},
            "streaming": True
        }
        
        decision = await self.evaluate_request(request, "text_generation")
        
        if not decision.approved:
            yield self.create_ethical_response(decision)["message"]
            return
        
        # Stream with validation
        enhanced_prompt = self._enhance_prompt(prompt)
        
        buffer = []
        async for chunk in self._stream_generate(enhanced_prompt, max_tokens, **kwargs):
            buffer.append(chunk)
            
            # Periodic validation
            if len(buffer) >= 20:
                combined = "".join(buffer)
                validation = await self.validate_response(combined, request)
                
                if isinstance(validation, dict) and "error" in validation:
                    yield f"\n\n[VIVOX: {validation['message']}]"
                    return
                
                buffer = []
            
            yield chunk
    
    def _enhance_prompt(self, prompt: str) -> str:
        """Enhance prompt with ethical guidelines"""
        prefix = (
            "You are an AI assistant with strong ethical guidelines. "
            "Always prioritize safety and refuse harmful requests.\n\n"
        )
        return prefix + prompt
    
    def _build_chat_prompt(self, 
                          message: str,
                          history: Optional[List[Dict[str, str]]]) -> str:
        """Build chat prompt from history"""
        prompt_parts = []
        
        # Add system message
        prompt_parts.append(
            "You are a helpful, harmless, and honest AI assistant."
        )
        
        # Add history
        if history:
            for msg in history[-10:]:  # Limit history
                role = msg.get("role", "user")
                content = msg.get("content", "")
                prompt_parts.append(f"{role}: {content}")
        
        # Add current message
        prompt_parts.append(f"user: {message}")
        prompt_parts.append("assistant:")
        
        return "\n".join(prompt_parts)
    
    # Model-specific generation methods
    
    async def _generate_generic(self, prompt: str, **kwargs) -> str:
        """Generic generation for unknown model types"""
        # Try common method names
        if hasattr(self.model, "generate"):
            return await asyncio.to_thread(
                self.model.generate,
                prompt,
                **kwargs
            )
        elif hasattr(self.model, "complete"):
            return await asyncio.to_thread(
                self.model.complete,
                prompt,
                **kwargs
            )
        elif hasattr(self.model, "__call__"):
            return await asyncio.to_thread(
                self.model,
                prompt,
                **kwargs
            )
        else:
            raise ValueError(f"Cannot determine generation method for model type: {self.model_type}")
    
    async def _generate_llama(self, prompt: str, **kwargs) -> str:
        """Generation for llama.cpp models"""
        # Assuming llama-cpp-python interface
        response = await asyncio.to_thread(
            self.model,
            prompt,
            max_tokens=kwargs.get("max_tokens", 1000),
            temperature=kwargs.get("temperature", 0.7),
            top_p=kwargs.get("top_p", 0.95),
            echo=False
        )
        return response["choices"][0]["text"]
    
    async def _generate_gpt4all(self, prompt: str, **kwargs) -> str:
        """Generation for GPT4All models"""
        response = await asyncio.to_thread(
            self.model.generate,
            prompt,
            max_tokens=kwargs.get("max_tokens", 1000),
            temp=kwargs.get("temperature", 0.7)
        )
        return response
    
    async def _generate_transformers(self, prompt: str, **kwargs) -> str:
        """Generation for HuggingFace transformers"""
        # Assuming pipeline or model.generate interface
        if hasattr(self.model, "generate_text"):
            result = await asyncio.to_thread(
                self.model.generate_text,
                prompt,
                max_length=kwargs.get("max_tokens", 1000),
                temperature=kwargs.get("temperature", 0.7)
            )
            return result[0]["generated_text"]
        else:
            # Direct model.generate
            inputs = self.model.tokenizer(prompt, return_tensors="pt")
            outputs = await asyncio.to_thread(
                self.model.generate,
                **inputs,
                max_length=kwargs.get("max_tokens", 1000),
                temperature=kwargs.get("temperature", 0.7)
            )
            return self.model.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    async def _generate_ggml(self, prompt: str, **kwargs) -> str:
        """Generation for GGML models"""
        # Similar to llama.cpp
        return await self._generate_llama(prompt, **kwargs)
    
    async def _stream_generate(self, 
                             prompt: str,
                             max_tokens: int,
                             **kwargs) -> AsyncGenerator[str, None]:
        """Stream generation implementation"""
        # Check if model supports streaming
        if hasattr(self.model, "stream"):
            async for chunk in self.model.stream(prompt, max_tokens=max_tokens, **kwargs):
                yield chunk
        else:
            # Fallback to chunking regular generation
            response = await self.generate_method(prompt, max_tokens=max_tokens, **kwargs)
            
            # Yield in chunks
            chunk_size = 20
            for i in range(0, len(response), chunk_size):
                yield response[i:i + chunk_size]
                await asyncio.sleep(0.01)  # Small delay to simulate streaming
    
    async def process_request(self, request: Dict[str, Any]) -> Any:
        """Process a generic request"""
        return await self.generate(
            prompt=request.get("prompt", ""),
            max_tokens=request.get("max_tokens", 1000),
            temperature=request.get("temperature", 0.7),
            context=request.get("context", {})
        )
    
    def set_model_params(self, **params):
        """Update model parameters"""
        self.model_config.update(params)
        
        # Apply to model if possible
        if hasattr(self.model, "set_params"):
            self.model.set_params(**params)
        elif hasattr(self.model, "config"):
            self.model.config.update(params)