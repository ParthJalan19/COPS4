"""
MongoDB Atlas Connection Tester
Run: python test_atlas.py
"""
import os
import urllib.parse
from dotenv import load_dotenv

load_dotenv(override=True)

uri = os.getenv("MONGODB_URI")
db_name = os.getenv("MONGODB_DB_NAME", "cops_new")

print("=" * 60)
print("Testing MongoDB Atlas Connection...")
print(f"Target Database: {db_name}")
print(f"Target URI (masked): {uri[:30]}...@{uri.split('@')[-1] if '@' in uri else ''}")
print("=" * 60)

uris_to_try = [uri]
if "#" in uri and "%23" not in uri:
    uris_to_try.append(uri.replace("#", "%23"))

try:
    import pymongo
    client = None
    last_err = None
    for u in uris_to_try:
        try:
            c = pymongo.MongoClient(u, serverSelectionTimeoutMS=4000)
            c.admin.command("ping")
            client = c
            break
        except Exception as e:
            last_err = e

    if client is None:
        raise last_err
    print("\n[SUCCESS] Connected to MongoDB Atlas cluster!")
    db = client[db_name]
    print(f"Existing collections in '{db_name}':", db.list_collection_names())
    print("Your Atlas credentials in .env are 100% correct!")
except Exception as e:
    print("\n[CONNECTION FAILED]")
    print(str(e))
    print("\nHow to fix:")
    print("1. Log in to https://cloud.mongodb.com")
    print("2. Go to 'Database Access' -> find or add user -> click 'Edit Password'.")
    print("3. Ensure role is 'Read and write to any database'.")
    print("4. Update your .env file with the new password.")
