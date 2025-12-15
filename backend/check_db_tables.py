
from sqlalchemy import inspect
from src.database import engine, init_db

def check_tables():
    print("Initializing database...")
    init_db()
    
    print("Connecting to database...")
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    print(f"Tables found: {tables}")
    if tables:
        print("✅ Tables exist.")
    else:
        print("❌ No tables found.")

if __name__ == "__main__":
    try:
        check_tables()
    except Exception as e:
        print(f"❌ Error checking tables: {e}")
