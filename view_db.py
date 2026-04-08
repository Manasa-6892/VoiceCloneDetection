import sqlite3

conn = sqlite3.connect("app/voice_clone_detection.db")
cursor = conn.cursor()

print("---- TABLES ----")
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
print(cursor.fetchall())

print("\n---- USERS ----")
cursor.execute("SELECT * FROM User")
for row in cursor.fetchall():
    print(row)

print("\n---- DETECTION LOGS ----")
cursor.execute("SELECT * FROM detection_log")
for row in cursor.fetchall():
    print(row)

conn.close()