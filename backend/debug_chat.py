
import sys
import os
import logging

# Ensure we can import from src
sys.path.append(os.getcwd())

from src.config import settings
from src.services.rag_service import rag_service
from src.services.qdrant_service import qdrant_service

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("debug")

def debug_chat():
    print("--- Debugging Chat ---")
    
    # 1. Check Qdrant Collection Status
    print("\n1. Checking Qdrant Collection...")
    try:
        collections = qdrant_service.client.get_collections()
        collection_names = [c.name for c in collections.collections]
        print(f"   Available Collections: {collection_names}")
        
        target_col = settings.vector_collection_name
        if target_col in collection_names:
            print(f"   ✅ Collection '{target_col}' exists.")
            # Check count
            count = qdrant_service.client.count(target_col)
            print(f"   📊 Items in collection: {count.count}")
        else:
            print(f"   ❌ Collection '{target_col}' DOES NOT EXIST.")
    except Exception as e:
        print(f"   ❌ Error checking collection: {e}")

    # 2. Try RAG Query
    print("\n2. Testing RAG Query...")
    print(f"DEBUG: Client type: {type(qdrant_service.client)}")
    print(f"DEBUG: Client dir: {dir(qdrant_service.client)}")
    try:
        result = rag_service.robust_query("hi")
        print(f"   Result: {result}")
        if result.get("error"):
            print(f"   ❌ Query returned error: {result['error']}")
        else:
            print("   ✅ Query successful.")
    except Exception as e:
        print(f"   ❌ Query crashed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_chat()
