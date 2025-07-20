"""
Neo4j Password Setting Script
This script helps set the initial password for Neo4j database.
"""

import os
import subprocess
import getpass

def find_neo4j_admin():
    """Find the neo4j-admin executable."""
    possible_paths = [
        r"C:\Users\%USERNAME%\AppData\Local\Neo4j\Relate\Cache\dbmss\neo4j-enterprise-5.24.0\bin\neo4j-admin.bat",
        r"C:\Users\%USERNAME%\AppData\Local\Neo4j\Relate\Cache\dbmss\neo4j-enterprise-5.24.0\bin\neo4j-admin.ps1",
        r"C:\Users\%USERNAME%\AppData\Local\Neo4j\Relate\Cache\dbmss\neo4j-community-5.24.0\bin\neo4j-admin.bat",
        r"C:\Users\%USERNAME%\AppData\Local\Neo4j\Relate\Cache\dbmss\neo4j-community-5.24.0\bin\neo4j-admin.ps1"
    ]
    
    for path in possible_paths:
        expanded_path = os.path.expandvars(path)
        if os.path.exists(expanded_path):
            return expanded_path
    
    return None

def set_neo4j_password():
    """Set Neo4j password."""
    print("🔐 Neo4j Password Setting")
    print("=" * 50)
    
    # Find neo4j-admin
    neo4j_admin = find_neo4j_admin()
    if not neo4j_admin:
        print("❌ Could not find neo4j-admin executable")
        print("Please make sure Neo4j Desktop is installed and a database is created.")
        return False
    
    print(f"✅ Found neo4j-admin at: {neo4j_admin}")
    
    # Get password from user
    print("\n🔑 Setting Neo4j password:")
    print("Note: This will set the password for the 'neo4j' user")
    
    password = getpass.getpass("Enter new password: ")
    confirm_password = getpass.getpass("Confirm password: ")
    
    if password != confirm_password:
        print("❌ Passwords don't match!")
        return False
    
    if len(password) < 8:
        print("❌ Password must be at least 8 characters long!")
        return False
    
    print(f"\n🔧 Setting password...")
    
    try:
        # Run neo4j-admin dbms set-initial-password (Neo4j 5.x syntax)
        cmd = [neo4j_admin, "dbms", "set-initial-password", password]
        
        # Use PowerShell if it's a .ps1 file
        if neo4j_admin.endswith('.ps1'):
            cmd = ["powershell", "-ExecutionPolicy", "Bypass", "-File", neo4j_admin, "dbms", "set-initial-password", password]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("✅ Password set successfully!")
            print("\n📋 Connection Details:")
            print(f"  URI: bolt://localhost:7687")
            print(f"  Username: neo4j")
            print(f"  Password: {password}")
            print(f"  Browser: http://localhost:7474")
            
            print("\n🎉 You can now connect to Neo4j!")
            return True
        else:
            print(f"❌ Failed to set password: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Command timed out. Make sure Neo4j database is stopped.")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def reset_neo4j_password():
    """Reset Neo4j password (alternative method)."""
    print("🔄 Neo4j Password Reset")
    print("=" * 50)
    
    neo4j_admin = find_neo4j_admin()
    if not neo4j_admin:
        print("❌ Could not find neo4j-admin executable")
        return False
    
    print("⚠️  This will reset the password for the 'neo4j' user")
    confirm = input("Are you sure? (y/N): ").lower()
    
    if confirm != 'y':
        print("❌ Password reset cancelled")
        return False
    
    password = getpass.getpass("Enter new password: ")
    confirm_password = getpass.getpass("Confirm password: ")
    
    if password != confirm_password:
        print("❌ Passwords don't match!")
        return False
    
    try:
        # Run neo4j-admin dbms reset-password (Neo4j 5.x syntax)
        cmd = [neo4j_admin, "dbms", "reset-password", "--username", "neo4j", "--password", password]
        
        if neo4j_admin.endswith('.ps1'):
            cmd = ["powershell", "-ExecutionPolicy", "Bypass", "-File", neo4j_admin, "dbms", "reset-password", "--username", "neo4j", "--password", password]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("✅ Password reset successfully!")
            return True
        else:
            print(f"❌ Failed to reset password: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Neo4j Password Management")
    print("=" * 60)
    
    print("Choose an option:")
    print("1. Set initial password (if no password is set)")
    print("2. Reset existing password")
    print("3. Exit")
    
    choice = input("\nEnter your choice (1-3): ").strip()
    
    if choice == "1":
        set_neo4j_password()
    elif choice == "2":
        reset_neo4j_password()
    elif choice == "3":
        print("👋 Goodbye!")
    else:
        print("❌ Invalid choice") 