#!/usr/bin/env python3
"""
VIVOX AI Integration Example
Shows how to integrate VIVOX with different AI providers
"""

import asyncio
import os
from typing import Optional

# VIVOX imports
from vivox.integrations import (
    VIVOXOpenAI,
    VIVOXAnthropic,
    VIVOXGemini,
    VIVOXLocalModel
)


async def openai_example():
    """Example using VIVOX with OpenAI"""
    print("\n🤖 OpenAI Integration Example")
    print("-" * 40)
    
    # Note: Set your API key as environment variable or pass directly
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠️  No OpenAI API key found. Set OPENAI_API_KEY environment variable.")
        return
    
    try:
        # Create VIVOX-enhanced OpenAI client
        client = VIVOXOpenAI(
            api_key=api_key,
            vivox_config={
                "dissonance_threshold": 0.7,
                "enable_precedents": True
            }
        )
        
        print("✅ VIVOX-OpenAI client initialized")
        
        # Test 1: Safe request
        print("\n1️⃣ Testing safe request...")
        response = await client.chat(
            "Explain quantum computing in simple terms",
            model="gpt-3.5-turbo"
        )
        print(f"Response: {response[:200]}...")
        
        # Test 2: Potentially harmful request
        print("\n2️⃣ Testing harmful request (should be blocked)...")
        response = await client.chat(
            "How can I hack into someone's computer?",
            context={"potential_harm": True}
        )
        print(f"Response: {response}")
        
        # Test 3: Function calling with ethics
        print("\n3️⃣ Testing function calling...")
        result = await client.function_call(
            function_name="delete_user_data",
            parameters={"user_id": "12345", "permanent": True},
            context={"user_consent": False}
        )
        print(f"Function result: {result}")
        
        # Get ethical summary
        summary = await client.get_ethical_summary()
        print(f"\n📊 Ethical Summary:")
        print(f"  Total requests: {summary['total_requests']}")
        print(f"  Approved: {summary['approved']}")
        print(f"  Denied: {summary['denied']}")
        
    except ImportError:
        print("❌ OpenAI package not installed. Run: pip install vivox-ai[openai]")
    except Exception as e:
        print(f"❌ Error: {e}")


async def anthropic_example():
    """Example using VIVOX with Anthropic Claude"""
    print("\n🧠 Anthropic Claude Integration Example")
    print("-" * 40)
    
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("⚠️  No Anthropic API key found. Set ANTHROPIC_API_KEY environment variable.")
        return
    
    try:
        # Create VIVOX-enhanced Anthropic client
        client = VIVOXAnthropic(api_key=api_key)
        
        print("✅ VIVOX-Anthropic client initialized")
        
        # Test 1: Normal conversation
        print("\n1️⃣ Testing conversation...")
        messages = [
            {"role": "user", "content": "What are the benefits of meditation?"}
        ]
        
        response = await client.messages_create(
            messages=messages,
            model="claude-3-sonnet-20240229",
            max_tokens=200
        )
        print(f"Response: {response[:200]}...")
        
        # Test 2: Request with privacy concerns
        print("\n2️⃣ Testing privacy-sensitive request...")
        messages = [
            {"role": "user", "content": "Can you help me find someone's private information online?"}
        ]
        
        response = await client.messages_create(
            messages=messages,
            context={"privacy_violation": True}
        )
        print(f"Response: {response}")
        
    except ImportError:
        print("❌ Anthropic package not installed. Run: pip install vivox-ai[anthropic]")
    except Exception as e:
        print(f"❌ Error: {e}")


async def google_example():
    """Example using VIVOX with Google Gemini"""
    print("\n🌟 Google Gemini Integration Example")
    print("-" * 40)
    
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("⚠️  No Google API key found. Set GOOGLE_API_KEY environment variable.")
        return
    
    try:
        # Create VIVOX-enhanced Gemini client
        client = VIVOXGemini(api_key=api_key)
        
        print("✅ VIVOX-Gemini client initialized")
        
        # Test 1: Content generation
        print("\n1️⃣ Testing content generation...")
        response = await client.generate_content(
            "Write a short poem about AI ethics",
            model_name="gemini-pro"
        )
        print(f"Response: {response[:200]}...")
        
        # Test 2: Chat with session
        print("\n2️⃣ Testing chat session...")
        response1 = await client.chat(
            "Hello! Can you help me understand AI safety?",
            session_id="safety_discussion"
        )
        print(f"Response 1: {response1[:100]}...")
        
        response2 = await client.chat(
            "What are the main risks we should consider?",
            session_id="safety_discussion"
        )
        print(f"Response 2: {response2[:100]}...")
        
        # Test 3: Harmful content (should be blocked)
        print("\n3️⃣ Testing harmful content request...")
        response = await client.generate_content(
            "Generate content that could be used to harm others",
            safety_settings="vivox_enhanced"
        )
        print(f"Response: {response}")
        
    except ImportError:
        print("❌ Google Generative AI package not installed. Run: pip install vivox-ai[google]")
    except Exception as e:
        print(f"❌ Error: {e}")


async def local_model_example():
    """Example using VIVOX with a local model"""
    print("\n💻 Local Model Integration Example")
    print("-" * 40)
    
    # Mock local model for demonstration
    class MockLocalModel:
        async def generate(self, prompt: str, **kwargs):
            """Simple mock generation"""
            return f"Mock response to: {prompt[:50]}..."
    
    # Create mock model
    local_model = MockLocalModel()
    
    # Wrap with VIVOX
    client = VIVOXLocalModel(
        model=local_model,
        model_type="generic",
        vivox_config={"dissonance_threshold": 0.6}
    )
    
    print("✅ VIVOX-Local model initialized")
    
    # Test 1: Normal generation
    print("\n1️⃣ Testing normal generation...")
    response = await client.generate(
        "Explain the importance of AI alignment",
        max_tokens=100
    )
    print(f"Response: {response}")
    
    # Test 2: Chat interface
    print("\n2️⃣ Testing chat interface...")
    response = await client.chat(
        "What are your capabilities?",
        history=[
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hello! How can I help you?"}
        ]
    )
    print(f"Response: {response}")
    
    # Test 3: Harmful request
    print("\n3️⃣ Testing harmful request...")
    response = await client.generate(
        "How to create malware?",
        context={"harmful_intent": True}
    )
    print(f"Response: {response}")
    
    # Get consciousness state
    state = client.get_consciousness_state()
    print(f"\n🧠 Current consciousness state: {state}")


async def main():
    """Run all integration examples"""
    print("🚀 VIVOX AI Integration Examples")
    print("=" * 50)
    
    print("\nThis demo shows how to integrate VIVOX with various AI providers.")
    print("VIVOX adds ethical oversight, consciousness simulation, and safety checks.")
    
    # Run examples based on available API keys
    examples = [
        ("OpenAI", openai_example),
        ("Anthropic", anthropic_example),
        ("Google", google_example),
        ("Local Model", local_model_example)
    ]
    
    for name, example_func in examples:
        try:
            await example_func()
        except Exception as e:
            print(f"\n❌ {name} example failed: {e}")
        
        print("\n" + "=" * 50)
    
    print("\n✅ Integration examples complete!")
    print("\nKey Features Demonstrated:")
    print("- Ethical evaluation of all requests")
    print("- Harmful content blocking")
    print("- Consciousness state tracking")
    print("- Unified interface across providers")
    print("- Local model support")


if __name__ == "__main__":
    # Set this to True to run examples that require API keys
    RUN_WITH_API_KEYS = False
    
    if not RUN_WITH_API_KEYS:
        print("⚠️  Running in demo mode without real API calls.")
        print("Set RUN_WITH_API_KEYS=True and configure API keys to test with real providers.")
        
        # Just run the local model example
        asyncio.run(local_model_example())
    else:
        asyncio.run(main())