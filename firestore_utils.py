import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime

# Load Firebase credentials
cred = credentials.Certificate("firebase_credentials.json")
firebase_admin.initialize_app(cred)

db = firestore.client()

def save_message(session_id, role, content, sources=None):
    """Save a chat message to Firestore."""
    doc_ref = db.collection("chat_history").document()
    doc_ref.set({
        "session_id": session_id,
        "role": role,
        "content": content,
        "sources": sources or [],
        "timestamp": datetime.utcnow()
    })

def load_messages(session_id):
    """Load chat messages from Firestore for a given session."""
    messages = []
    docs = db.collection("chat_history")\
        .where("session_id", "==", session_id)\
        .order_by("timestamp")\
        .stream()
    for doc in docs:
        messages.append(doc.to_dict())
    return messages
