import asyncio
from datetime import datetime, timezone
import socketio
from aiohttp import web
from aiohttp_middlewares import cors_middleware
from chatbot import generate_response, generate_welcome_message
from dotenv import load_dotenv
load_dotenv()

from datetime import timedelta

def now_iso():
    # Returns the current epoch timestamp as an integer
    return int(datetime.now().timestamp())

# Create a new Socket.IO server instance with asyncio support
sio = socketio.AsyncServer(cors_allowed_origins="*")  # Allow all origins for Socket.IO

# Create an instance of the aiohttp web application
app = web.Application(
    middlewares=[cors_middleware(allow_all=True)]
)  # Explicitly allow all CORS

# Attach the Socket.IO server to the app
sio.attach(app)


async def index(request):
    return web.FileResponse("index.html")


app.router.add_get("/", index)

# Store chat history per session ID
chat_history = {}
client_sessions = {}  # Maps socket sid to persistent session_id
active_timers = {}    # Maps session_id to active asyncio follow-up timers

async def followup_timer(session_id, sid, timeout=240):
    try:
        await asyncio.sleep(timeout)
        # Timer finished without being cancelled!
        if session_id in chat_history:
            form_already_submitted = any("form_submitted" in m for m in chat_history[session_id])
            if form_already_submitted:
                return  # Don't ask if the form is already filled up
                
        msg = "Hey, are you still there? Happy to help if you have more questions."
        ts = now_iso()
        if session_id in chat_history:
            chat_history[session_id].append({"bot": msg, "timestamp": ts})
        await sio.emit("message", {"session_id": session_id, "message": msg, "timestamp": ts}, room=sid)
    except asyncio.CancelledError:
        # User replied in time, so timer was cancelled.
        pass


# Event handler for client connection
@sio.event
async def connect(sid, environ):
    print("New client connected:", sid)
    # Do NOT emit anything here.
    # The frontend immediately emits "get_chat_history" after connecting.
    # We handle session creation/resumption there to avoid overwriting the cookie.


# Event handler for client disconnection
@sio.event
async def disconnect(sid):
    print("Client disconnected:", sid)
    client_sessions.pop(sid, None)


# Event handler for receiving messages from clients
@sio.on("message")  # "message" is the custom event that the client will emit
async def handle_message(sid, message):
    import uuid
    print(f"Received query from {sid}: {message}")
    try:
        session_id = None
        # If the frontend sends a dict, extract the string message and session_id
        if isinstance(message, dict):
            user_msg = message.get("message", str(message))
            if "session_id" in message and message["session_id"]:
                session_id = message["session_id"]
        else:
            user_msg = str(message)

        # If they didn't provide a session_id, fallback to the one tied to this socket connection
        if not session_id:
            session_id = client_sessions.get(sid)
            if not session_id:
                session_id = str(uuid.uuid4())
                
        # Update the active session for this socket
        client_sessions[sid] = session_id
        
        # Cancel any active follow-up timer for this session since the user just replied
        if session_id in active_timers:
            active_timers[session_id].cancel()
            del active_timers[session_id]

        # Generate a parent request ID for this workflow interaction
        parent_request_id = f"chat_{uuid.uuid4().hex[:12]}"

        # Initialize history if session_id is new
        if session_id not in chat_history:
            welcome_msg = await generate_welcome_message(
                parent_request_id=parent_request_id,
                session_id=session_id,
            )
            chat_history[session_id] = [{"bot": welcome_msg, "timestamp": now_iso()}]

        chat_history[session_id].append({"user": user_msg, "timestamp": now_iso()})

        # Count total messages (user + bot) in the current session
        total_messages = len(chat_history[session_id])  # includes both user and bot entries
        form_already_shown = any("form" in m for m in chat_history[session_id])

        # After 10 total messages (approximately 5 exchanges) and if the form hasn't been shown yet,
        # directly show the form instead of nudging the bot
        actual_query = user_msg
        show_form = False
        if total_messages >= 10 and not form_already_shown:
            show_form = True

        response_text = await generate_response(
            actual_query,
            chat_history[session_id][:-1],
            parent_request_id=parent_request_id,
            session_id=session_id,
        )

        if "[SHOW_FORM]" in response_text:
            show_form = True
            response_text = response_text.replace("[SHOW_FORM]", "").strip()

        # Log the response in chat history
        bot_ts = now_iso()
        chat_history[session_id].append({"bot": response_text, "timestamp": bot_ts})
        if show_form:
            chat_history[session_id].append({"form": "true", "timestamp": bot_ts})
            
        print(response_text)

        # Send the generated response back to the client along with the session_id
        emit_data = {"session_id": session_id, "message": response_text, "timestamp": bot_ts}
        await sio.emit("message", emit_data, room=sid)
        
        if show_form:
            await sio.emit("message", {"session_id": session_id, "form": "true", "timestamp": bot_ts}, room=sid)

        print("Response Successfully Sent")
        
        # Start a new follow-up timer after sending the response
        active_timers[session_id] = asyncio.create_task(followup_timer(session_id, sid))
    except Exception as e:
        import traceback
        print(f"Error while processing message from {sid}: {e}\n{traceback.format_exc()}")
        user_error_message = "We are having trouble reaching our servers right now. Please try again later."
        active_sid = locals().get('session_id')
        if active_sid and active_sid in chat_history:
            chat_history[active_sid].append({"bot": user_error_message, "timestamp": now_iso()})
        await sio.emit("message", {"session_id": active_sid, "message": user_error_message, "timestamp": now_iso()}, room=sid)


# Event handler to send entire chat history on demand
@sio.on("get_chat_history")
async def handle_get_chat_history(sid, *args):
    import uuid
    requested_id = None
    if args:
        if isinstance(args[0], dict) and "session_id" in args[0]:
            requested_id = args[0]["session_id"].strip()
        elif isinstance(args[0], str):
            requested_id = args[0].strip()

    # Determine whether this is a resume or a new session
    if requested_id and requested_id in chat_history:
        # --- RESUME: known session found ---
        session_id = requested_id
        client_sessions[sid] = session_id
        print(f"Resuming session {session_id} for connection {sid}")
        
        history = chat_history[session_id]
        print(history)
        await sio.emit("chat_history", history, room=sid)
    else:
        # --- NEW SESSION: unknown / blank / first visit ---
        session_id = str(uuid.uuid4())
        client_sessions[sid] = session_id
        page_name = args[0].get("page_name") if args and isinstance(args[0], dict) else None
        parent_request_id = f"welcome_{uuid.uuid4().hex[:12]}"
        welcome_msg = await generate_welcome_message(
            page_name=page_name,
            parent_request_id=parent_request_id,
            session_id=session_id,
        )
        welcome_ts = now_iso()
        chat_history[session_id] = [{"bot": welcome_msg, "timestamp": welcome_ts}]
        print(f"New session {session_id} for connection {sid}")
        await sio.emit("chat_history", chat_history[session_id], room=sid)


# Event handler for form submission
@sio.on("form_submitted")
async def handle_form_submitted(sid, *args):
    session_id = client_sessions.get(sid)
    if not session_id:
        return
        
    msg = "Thank you for filling up the details, our team will reach out to you soon. If you have anymore doubts, you can ask me"
    ts = now_iso()
    
    if session_id in chat_history:
        chat_history[session_id].append({"bot": msg, "timestamp": ts})
        chat_history[session_id].append({"form_submitted": "true", "timestamp": ts})
        
    await sio.emit("message", {"session_id": session_id, "message": msg, "timestamp": ts}, room=sid)



# Event handler to force create a new chat session
@sio.on("new_chat")
async def handle_new_chat(sid, *args):
    import uuid
    # Generate a brand new session ID
    session_id = str(uuid.uuid4())
    client_sessions[sid] = session_id
    
    page_name = args[0].get("page_name") if args and isinstance(args[0], dict) else None
    parent_request_id = f"newchat_{uuid.uuid4().hex[:12]}"
    welcome_msg = await generate_welcome_message(
        page_name=page_name,
        parent_request_id=parent_request_id,
        session_id=session_id,
    )
    welcome_ts = now_iso()
    chat_history[session_id] = [{"bot": welcome_msg, "timestamp": welcome_ts}]
    print(f"Force created new session {session_id} for connection {sid}")
    
    # Send the new session ID and welcome message back to client
    await sio.emit("message", {"session_id": session_id, "message": welcome_msg, "timestamp": welcome_ts}, room=sid)
    # Send the fresh chat history so the frontend UI clears out old messages
    await sio.emit("chat_history", chat_history[session_id], room=sid)


# Run the aiohttp web application
if __name__ == "__main__":
    # import ssl

    # ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    # ssl_context.load_cert_chain(
    #     certfile="/etc/ssl/certs/fullchain.pem",
    #     keyfile="/etc/ssl/private/multiqos.com.key",
    # )

    # web.run_app(app, host="0.0.0.0", port=8877, ssl_context=ssl_context)
    web.run_app(app, host="0.0.0.0", port=8877)
