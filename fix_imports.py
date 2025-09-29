#!/usr/bin/env python3
"""
Quick fix script for import issues
Run this to test and fix import problems
"""

import sys
import os
import subprocess

def fix_imports():
    """Fix import issues by installing missing dependencies"""
    print("🔧 Fixing import issues...")
    
    # Add src directory to Python path
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
    
    try:
        # Test if we can import the modules
        from obsidian_chunker import ObsidianChunker
        from simple_raganything import SimpleRAGAnything
        print("✅ All imports working!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("🔄 Installing missing dependencies...")
        
        # Install missing packages
        packages = ['frontmatter', 'dataclasses']
        for package in packages:
            try:
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
                print(f"✅ Installed {package}")
            except subprocess.CalledProcessError:
                print(f"❌ Failed to install {package}")
        
        # Test again
        try:
            from obsidian_chunker import ObsidianChunker
            from simple_raganything import SimpleRAGAnything
            print("✅ All imports working after fix!")
            return True
        except ImportError as e2:
            print(f"❌ Still having import issues: {e2}")
            return False

if __name__ == "__main__":
    fix_imports()
