
import sys
import os
sys.path.append(os.getcwd())

from src.services.qdrant_service import qdrant_service
from src.services.cohere_service import cohere_service
from src.config import settings

def debug_retrieval():
    print(f"--- Debugging Retrieval for collection: {settings.vector_collection_name} ---")
    
    # 1. Check Count
    try:
        count = qdrant_service.client.count(settings.vector_collection_name)
        print(f"📊 Total Points: {count.count}")
    except Exception as e:
        print(f"❌ Error counting points: {e}")
        return

    # 2. Test Specific Queries
    test_queries = [
        "Physical_AI_Humanoid_Robotics",
        "details of this book",
        "intro"
    ]

    for query in test_queries:
        print(f"\n🔎 Testing Query: '{query}'")
        try:
            # Generate embedding
            embedding = cohere_service.generate_embedding(query)
            
            # Raw Search
            results = qdrant_service.client.query_points(
                collection_name=settings.vector_collection_name,
                query=embedding,
                limit=3
            ).points

            if not results:
                print("   ⚠️ No results found.")
            
            for hit in results:
                print(f"   - Score: {hit.score:.4f}")
                print(f"     Content: {hit.payload.get('content', '')[:100]}...")
                
        except Exception as e:
            print(f"   ❌ Search failed: {e}")

if __name__ == "__main__":
    debug_retrieval()
