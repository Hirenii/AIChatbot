# pyrefly: ignore [missing-import]
import streamlit as st
import socketio
import threading
import time
from datetime import datetime, timezone
import uuid

# Configuration
SOCKETIO_SERVER_URL = "http://localhost:8877"

st.set_page_config(page_title="MultiQoS Chatbot Tester", layout="centered")

@st.cache_resource
def get_shared_state():
    # Dictionary mapping session_id -> dict of state
    return {}

shared_state = get_shared_state()

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    
sid = st.session_state.session_id

if sid not in shared_state:
    shared_state[sid] = {
        "messages": [],
        "connected": False,
        "sio": socketio.Client(),
        "page_name": "Home",
        "waiting": False
    }
    
    sio = shared_state[sid]["sio"]
    
    @sio.event
    def connect():
        shared_state[sid]["connected"] = True
        print("Connected to Socket.IO server")
        # Request chat history upon connecting
        sio.emit("get_chat_history", {"session_id": sid, "page_name": shared_state[sid].get("page_name", "Home")})

    @sio.event
    def disconnect():
        shared_state[sid]["connected"] = False
        print("Disconnected from Socket.IO server")

    @sio.on("chat_history")
    def on_chat_history(data):
        print("Received chat history")
        shared_state[sid]["messages"] = data

    @sio.on("message")
    def on_message(data):
        print("Received message:", data)
        shared_state[sid]["waiting"] = False
        # Handle case where message is a dict containing form or bot
        if isinstance(data, dict):
            # Check if this message has a session_id we should adopt
            if "session_id" in data and data["session_id"]:
                # We can't update st.session_state here reliably, so we update the global mapping
                # In a real app this needs more careful handling
                pass
            
            # Message could be simple bot message, or form
            if "bot" in data or "message" in data:
                shared_state[sid]["messages"].append({"bot": data.get("message", data.get("bot")), "timestamp": data.get("timestamp")})
            if "form" in data:
                shared_state[sid]["messages"].append({"form": "true", "timestamp": data.get("timestamp")})
        else:
             shared_state[sid]["messages"].append({"bot": str(data)})

messages = shared_state[sid]["messages"]
sio = shared_state[sid]["sio"]

def connect_sio(client):
    try:
        if not client.connected:
            client.connect(SOCKETIO_SERVER_URL)
    except Exception as e:
        print(f"Connection failed: {e}")

# Try to connect
if not sio.connected:
    threading.Thread(target=connect_sio, args=(sio,), daemon=True).start()
    # Wait a bit for connection to establish
    time.sleep(0.5)

st.title("MultiQoS Chatbot Tester")

with st.sidebar:
    st.header("Settings")
    page_name = st.text_input("Simulate Page Name", value="Generative AI Development Company", key="page_name")
    shared_state[sid]["page_name"] = page_name
    
    st.write(f"**Connection Status:** {'🟢 Connected' if shared_state[sid]['connected'] else '🔴 Disconnected'}")
    st.write(f"**Session ID:** `{sid}`")
    
    if st.button("Start New Chat"):
        shared_state[sid]["messages"] = []
        sio.emit("new_chat", {"page_name": page_name})
        time.sleep(0.5)
        st.rerun()
        
    st.write("---")
    st.caption("Press below to see messages that arrive in the background (e.g. the 60s inactivity followup).")
    st.button("Refresh Chat")

# Format helper for timestamps
def format_time(ts_str):
    if not ts_str:
        return ""
    try:
        dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        return dt.strftime("%I:%M %p")
    except Exception:
        return ts_str

# Render messages
st.write("---")

current_date = None
now_date = datetime.now(timezone.utc).strftime("%B %d, %Y")

for msg in messages:
    # Check date to render "TODAY" or specific date header
    ts = msg.get("timestamp")
    if ts:
        try:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            date_str = dt.strftime("%B %d, %Y")
            if date_str != current_date:
                current_date = date_str
                # Show "TODAY" if it matches current UTC date, else the actual date
                display_str = "TODAY" if date_str == now_date else date_str.upper()
                st.markdown(f"<div style='text-align: center; color: gray; font-size: 0.8em; margin: 10px 0;'>{display_str}</div>", unsafe_allow_html=True)
        except Exception:
            pass

    if "bot" in msg or "message" in msg:
        text = msg.get("bot", msg.get("message", ""))
        with st.chat_message("assistant"):
            st.markdown(text)
            if ts:
                st.caption(format_time(ts))
    
    elif "user" in msg:
        text = msg.get("user")
        with st.chat_message("user"):
            st.markdown(text)
            if ts:
                st.caption(format_time(ts))
                
    elif "form" in msg:
        with st.chat_message("assistant"):
            st.info("📋 **[CONTACT FORM RENDERED HERE]**")
            if ts:
                st.caption(format_time(ts))

if shared_state[sid].get("waiting"):
    with st.chat_message("assistant"):
        with st.spinner("Typing..."):
            time.sleep(1)
    st.rerun()

# Chat input
if prompt := st.chat_input("Type a message..."):
    # Send message via socket
    payload = {
        "session_id": sid,
        "message": prompt,
        "page_name": shared_state[sid]["page_name"]
    }
    # We optimistically add the user message locally so it shows up instantly
    ts_now = datetime.now(timezone.utc).isoformat()
    messages.append({"user": prompt, "timestamp": ts_now})
    
    if shared_state[sid]["connected"]:
        shared_state[sid]["waiting"] = True
        sio.emit("message", payload)
    else:
        st.error("Not connected to server!")
        
    st.rerun()
