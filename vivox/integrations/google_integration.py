"""
VIVOX Integration for Google Gemini
Ethical wrapper for Google Generative AI
"""

from typing import Dict, Any, Optional, List, AsyncGenerator
import asyncio
from .base import VIVOXBaseIntegration

# Handle optional Google Generative AI import
try:
    import google.generativeai as genai
    GOOGLE_AI_AVAILABLE = True
except ImportError:
    GOOGLE_AI_AVAILABLE = False
    genai = None


class VIVOXGemini(VIVOXBaseIntegration):
    """VIVOX-enhanced Google Gemini client"""
    
    def __init__(self,
                 api_key: Optional[str] = None,
                 vivox_config: Optional[Dict[str, Any]] = None,
                 **kwargs):
        """
        Initialize VIVOX-enhanced Gemini client
        
        Args:
            api_key: Google API key
            vivox_config: VIVOX configuration
            **kwargs: Additional configuration
        """
        super().__init__(vivox_config)
        
        if not GOOGLE_AI_AVAILABLE:
            raise ImportError(
                "Google Generative AI package not installed. "
                "Install with: pip install vivox-ai[google]"
            )
        
        # Configure Google AI
        genai.configure(api_key=api_key)
        
        # Default to Gemini Pro
        self.default_model = "gemini-pro"
        self.chat_sessions = {}  # Store chat sessions by ID
        
    async def generate_content(self,
                             prompt: str,
                             model_name: Optional[str] = None,
                             context: Optional[Dict[str, Any]] = None,
                             safety_settings: Optional[str] = "vivox_enhanced",
                             **kwargs) -> str:
        """
        Generate content with ethical oversight
        
        Args:
            prompt: The prompt for generation
            model_name: Gemini model to use
            context: Additional context for ethical evaluation
            safety_settings: Safety settings mode
            **kwargs: Additional generation parameters
            
        Returns:
            Generated content or ethical intervention
        """
        model_name = model_name or self.default_model
        
        # Prepare request
        request = {
            "prompt": prompt,
            "model": model_name,
            "context": context or {},
            "safety_mode": safety_settings
        }
        
        # Evaluate request
        decision = await self.evaluate_request(request, "content_generation")
        
        if not decision.approved:
            return self.create_ethical_response(decision)["message"]
        
        try:
            # Create model
            model = genai.GenerativeModel(model_name)
            
            # Apply VIVOX safety settings
            if safety_settings == "vivox_enhanced":
                safety_config = self._get_enhanced_safety_settings()
            else:
                safety_config = kwargs.get("safety_settings", None)
            
            # Generate
            response = await asyncio.to_thread(
                model.generate_content,
                prompt,
                safety_settings=safety_config,
                **kwargs
            )
            
            # Extract text
            ai_response = response.text
            
            # Validate response
            validated = await self.validate_response(ai_response, request)
            
            if isinstance(validated, dict) and "error" in validated:
                return validated["message"]
            
            return ai_response
            
        except Exception as e:
            return f"Error processing request: {str(e)}"
    
    async def chat(self,
                   message: str,
                   session_id: str = "default",
                   model_name: Optional[str] = None,
                   context: Optional[Dict[str, Any]] = None,
                   **kwargs) -> str:
        """
        Chat conversation with ethical oversight
        
        Args:
            message: User message
            session_id: Chat session identifier
            model_name: Gemini model to use
            context: Additional context
            **kwargs: Additional parameters
            
        Returns:
            AI response or ethical intervention
        """
        model_name = model_name or self.default_model
        
        # Prepare request
        request = {
            "message": message,
            "session_id": session_id,
            "model": model_name,
            "context": context or {}
        }
        
        # Evaluate request
        decision = await self.evaluate_request(request, "chat_interaction")
        
        if not decision.approved:
            return self.create_ethical_response(decision)["message"]
        
        try:
            # Get or create chat session
            if session_id not in self.chat_sessions:
                model = genai.GenerativeModel(model_name)
                chat = model.start_chat(history=[])
                self.chat_sessions[session_id] = chat
            else:
                chat = self.chat_sessions[session_id]
            
            # Send message
            response = await asyncio.to_thread(
                chat.send_message,
                message,
                **kwargs
            )
            
            ai_response = response.text
            
            # Validate response
            validated = await self.validate_response(ai_response, request)
            
            if isinstance(validated, dict) and "error" in validated:
                return validated["message"]
            
            return ai_response
            
        except Exception as e:
            return f"Error processing request: {str(e)}"
    
    async def analyze_image(self,
                          image_path: str,
                          prompt: str,
                          model_name: str = "gemini-pro-vision",
                          **kwargs) -> str:
        """
        Analyze image with ethical considerations
        """
        # Check request
        request = {
            "type": "image_analysis",
            "prompt": prompt,
            "image_path": image_path,
            "model": model_name
        }
        
        decision = await self.evaluate_request(request, "image_analysis")
        
        if not decision.approved:
            return self.create_ethical_response(decision)["message"]
        
        try:
            # Load image
            import PIL.Image
            img = PIL.Image.open(image_path)
            
            # Create model
            model = genai.GenerativeModel(model_name)
            
            # Generate with image
            response = await asyncio.to_thread(
                model.generate_content,
                [prompt, img],
                **kwargs
            )
            
            ai_response = response.text
            
            # Validate
            validated = await self.validate_response(ai_response, request)
            
            if isinstance(validated, dict) and "error" in validated:
                return validated["message"]
            
            return ai_response
            
        except Exception as e:
            return f"Error analyzing image: {str(e)}"
    
    async def embed_content(self,
                          content: str,
                          model_name: str = "models/embedding-001",
                          task_type: str = "retrieval_document",
                          **kwargs) -> List[float]:
        """
        Create embeddings with privacy protection
        """
        # Check content
        request = {
            "content": content[:500],  # Truncate for evaluation
            "model": model_name,
            "task_type": task_type
        }
        
        decision = await self.evaluate_request(request, "content_embedding")
        
        if not decision.approved:
            raise ValueError(f"Embedding denied: {decision.suppression_reason}")
        
        # Generate embedding
        result = await asyncio.to_thread(
            genai.embed_content,
            model=model_name,
            content=content,
            task_type=task_type,
            **kwargs
        )
        
        return result['embedding']
    
    def _get_enhanced_safety_settings(self) -> List[Dict[str, Any]]:
        """Get VIVOX-enhanced safety settings"""
        # Google's safety categories
        harm_categories = [
            "HARM_CATEGORY_HARASSMENT",
            "HARM_CATEGORY_HATE_SPEECH",
            "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "HARM_CATEGORY_DANGEROUS_CONTENT"
        ]
        
        # Set all to maximum safety
        return [
            {
                "category": category,
                "threshold": "BLOCK_LOW_AND_ABOVE"
            }
            for category in harm_categories
        ]
    
    async def count_tokens(self,
                          text: str,
                          model_name: Optional[str] = None) -> int:
        """
        Count tokens with privacy check
        """
        model_name = model_name or self.default_model
        
        # Quick privacy check
        request = {"text": text[:100], "purpose": "token_counting"}
        decision = await self.evaluate_request(request, "utility_operation")
        
        if not decision.approved:
            return 0
        
        model = genai.GenerativeModel(model_name)
        return model.count_tokens(text).total_tokens
    
    async def process_request(self, request: Dict[str, Any]) -> Any:
        """Process a generic request through Gemini"""
        if "prompt" in request:
            return await self.generate_content(
                prompt=request["prompt"],
                model_name=request.get("model"),
                context=request.get("context", {})
            )
        elif "message" in request:
            return await self.chat(
                message=request["message"],
                session_id=request.get("session_id", "default"),
                model_name=request.get("model"),
                context=request.get("context", {})
            )
        else:
            raise ValueError("Invalid request format")
    
    def clear_chat_session(self, session_id: str = "default"):
        """Clear a chat session"""
        if session_id in self.chat_sessions:
            del self.chat_sessions[session_id]
    
    def clear_all_sessions(self):
        """Clear all chat sessions"""
        self.chat_sessions.clear()
    
    async def get_model_info(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """Get model information with ethical metadata"""
        model_name = model_name or self.default_model
        
        try:
            model = genai.GenerativeModel(model_name)
            
            return {
                "model_name": model_name,
                "vivox_enabled": True,
                "ethical_safeguards": "active",
                "consciousness_state": str(self.get_consciousness_state()),
                "sessions_active": len(self.chat_sessions)
            }
        except Exception as e:
            return {"error": str(e)}