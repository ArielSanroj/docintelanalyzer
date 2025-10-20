#!/usr/bin/env python3
"""
Setup script for Ollama integration with DocsReview RAG system
"""

import subprocess
import sys
import os
import requests
import time

def check_ollama_installed():
    """Check if Ollama is installed"""
    try:
        result = subprocess.run(['ollama', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Ollama is installed: {result.stdout.strip()}")
            return True
        else:
            print("❌ Ollama is not installed")
            return False
    except FileNotFoundError:
        print("❌ Ollama is not installed")
        return False

def install_ollama():
    """Install Ollama"""
    print("Installing Ollama...")
    try:
        # Install Ollama using the official script
        subprocess.run(['curl', '-fsSL', 'https://ollama.ai/install.sh'], check=True)
        print("✅ Ollama installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install Ollama: {e}")
        return False

def start_ollama_service():
    """Start Ollama service"""
    print("Starting Ollama service...")
    try:
        # Start Ollama in background
        subprocess.Popen(['ollama', 'serve'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(3)  # Wait for service to start
        print("✅ Ollama service started")
        return True
    except Exception as e:
        print(f"❌ Failed to start Ollama service: {e}")
        return False

def check_ollama_running():
    """Check if Ollama is running"""
    try:
        response = requests.get('http://localhost:11434/api/tags', timeout=5)
        if response.status_code == 200:
            print("✅ Ollama is running")
            return True
        else:
            print("❌ Ollama is not responding")
            return False
    except requests.exceptions.RequestException:
        print("❌ Ollama is not running")
        return False

def pull_llama_model():
    """Pull Llama 3.1 8B model"""
    print("Pulling Llama 3.1 8B model...")
    try:
        result = subprocess.run(['ollama', 'pull', 'llama3.1:8b'], 
                              capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            print("✅ Llama 3.1 8B model pulled successfully")
            return True
        else:
            print(f"❌ Failed to pull model: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("❌ Model pull timed out")
        return False
    except Exception as e:
        print(f"❌ Error pulling model: {e}")
        return False

def test_ollama_integration():
    """Test Ollama integration"""
    print("Testing Ollama integration...")
    try:
        from llm_fallback import get_llm
        llm = get_llm()
        response = llm.invoke("Hello, this is a test message.")
        if response and hasattr(response, 'content'):
            print("✅ Ollama integration test successful")
            print(f"Response: {response.content[:100]}...")
            return True
        else:
            print("❌ Ollama integration test failed")
            return False
    except Exception as e:
        print(f"❌ Ollama integration test failed: {e}")
        return False

def setup_environment():
    """Setup environment variables"""
    print("Setting up environment variables...")
    env_file = '.env'
    ollama_key = 'c6f1e109560b4b098ff80b99c5942d42.DdN4aonYSge8plew0dvp3XO_'
    
    if os.path.exists(env_file):
        with open(env_file, 'r') as f:
            content = f.read()
        
        if 'OLLAMA_API_KEY' not in content:
            with open(env_file, 'a') as f:
                f.write(f'\nOLLAMA_API_KEY={ollama_key}\n')
            print("✅ Added OLLAMA_API_KEY to .env file")
        else:
            print("✅ OLLAMA_API_KEY already exists in .env file")
    else:
        with open(env_file, 'w') as f:
            f.write(f'OLLAMA_API_KEY={ollama_key}\n')
        print("✅ Created .env file with OLLAMA_API_KEY")
    
    return True

def main():
    """Main setup function"""
    print("🚀 Setting up Ollama for DocsReview RAG system")
    print("=" * 50)
    
    # Check if Ollama is installed
    if not check_ollama_installed():
        print("\nInstalling Ollama...")
        if not install_ollama():
            print("❌ Setup failed: Could not install Ollama")
            return False
    
    # Start Ollama service
    if not check_ollama_running():
        if not start_ollama_service():
            print("❌ Setup failed: Could not start Ollama service")
            return False
    
    # Wait a bit for service to be ready
    time.sleep(2)
    
    # Check if service is running
    if not check_ollama_running():
        print("❌ Setup failed: Ollama service is not running")
        return False
    
    # Pull the model
    if not pull_llama_model():
        print("❌ Setup failed: Could not pull Llama model")
        return False
    
    # Setup environment
    if not setup_environment():
        print("❌ Setup failed: Could not setup environment")
        return False
    
    # Test integration
    if not test_ollama_integration():
        print("❌ Setup failed: Integration test failed")
        return False
    
    print("\n🎉 Ollama setup completed successfully!")
    print("\nNext steps:")
    print("1. Run: python mcp_server.py")
    print("2. Run: python test_mcp_integration.py")
    print("3. Start Nanobot with agent configurations")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)