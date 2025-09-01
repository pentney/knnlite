#!/usr/bin/env python3
"""
Setup script for KNNLite Database Integration

This script helps set up the database environment for KNNLite.
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required Python packages."""
    print("Installing Python requirements...")
    try:
        subprocess.check_call([
            sys.executable, '-m', 'pip', 'install', '-r', 'requirements_database.txt'
        ])
        print("✓ Python packages installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install Python packages: {e}")
        return False
    return True

def check_postgresql():
    """Check if PostgreSQL is available."""
    print("Checking PostgreSQL availability...")
    try:
        result = subprocess.run(['psql', '--version'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✓ PostgreSQL found: {result.stdout.strip()}")
            return True
        else:
            print("✗ PostgreSQL not found")
            return False
    except FileNotFoundError:
        print("✗ PostgreSQL not found in PATH")
        return False

def create_test_database():
    """Create a test database for KNNLite."""
    print("Creating test database...")
    
    # Database parameters
    db_params = {
        'host': 'localhost',
        'port': 5432,
        'database': 'postgres',  # Connect to default database first
        'user': 'postgres',
        'password': 'password'
    }
    
    try:
        import psycopg2
        
        # Connect to PostgreSQL
        conn = psycopg2.connect(**db_params)
        conn.autocommit = True
        
        with conn.cursor() as cursor:
            # Create test database
            cursor.execute("CREATE DATABASE knnlite_test;")
            print("✓ Test database 'knnlite_test' created")
            
            # Connect to test database
            conn.close()
            db_params['database'] = 'knnlite_test'
            conn = psycopg2.connect(**db_params)
            conn.autocommit = True
            
            with conn.cursor() as cursor:
                # Install pgvector extension
                cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                print("✓ pgvector extension installed")
        
        conn.close()
        return True
        
    except ImportError:
        print("✗ psycopg2 not installed. Run: pip install psycopg2-binary")
        return False
    except Exception as e:
        print(f"✗ Failed to create test database: {e}")
        print("Make sure PostgreSQL is running and accessible")
        return False

def run_test():
    """Run a simple test to verify everything works."""
    print("Running test...")
    try:
        # Import and test the database module
        from knnlite_database import DatabaseHNSWBuilder, create_sample_table
        
        db_params = {
            'host': 'localhost',
            'port': 5432,
            'database': 'knnlite_test',
            'user': 'postgres',
            'password': 'password'
        }
        
        # Create sample table
        create_sample_table(db_params, 'test_table', 100, 64)
        
        # Test HNSW builder
        with DatabaseHNSWBuilder(db_params) as builder:
            classifier = builder.build_hnsw_index('test_table', 'vector', 'label')
            print(f"✓ HNSW index built with {classifier.num_nodes} nodes")
        
        print("✓ All tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False

def main():
    """Main setup function."""
    print("KNNLite Database Integration Setup")
    print("=" * 40)
    
    success = True
    
    # Install requirements
    if not install_requirements():
        success = False
    
    # Check PostgreSQL
    if not check_postgresql():
        print("\nTo install PostgreSQL:")
        print("Ubuntu/Debian: sudo apt-get install postgresql postgresql-contrib")
        print("macOS: brew install postgresql")
        print("Windows: Download from https://www.postgresql.org/download/")
        success = False
    
    # Create test database
    if success and not create_test_database():
        success = False
    
    # Run test
    if success and not run_test():
        success = False
    
    print("\n" + "=" * 40)
    if success:
        print("✓ Setup completed successfully!")
        print("\nYou can now run:")
        print("  python3 database_example.py")
    else:
        print("✗ Setup failed. Please check the errors above.")
        print("\nManual setup steps:")
        print("1. Install PostgreSQL and pgvector extension")
        print("2. Create a database for testing")
        print("3. Install Python requirements: pip install -r requirements_database.txt")
        print("4. Update database connection parameters in the scripts")

if __name__ == "__main__":
    main()
