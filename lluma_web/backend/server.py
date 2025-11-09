from flask import Flask, jsonify, send_from_directory
from flask_socketio import SocketIO, emit
import os
import time
import json
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Construct the absolute path for the static folder
script_dir = os.path.dirname(os.path.abspath(__file__))
static_folder_path = os.path.abspath(os.path.join(script_dir, '..', 'frontend', 'build'))

app = Flask(__name__, static_folder=static_folder_path, static_url_path='/')
socketio = SocketIO(app, cors_allowed_origins="*")

# In-memory state
latest_turn_data = None
latest_turn_number = -1

def get_latest_turn_files(logs_dir, captures_dir):
    """Finds the latest turn number and associated files."""
    try:
        log_files = [f for f in os.listdir(logs_dir) if f.startswith('turn_') and f.endswith('.json')]
        if not log_files:
            return -1, None, None

        latest_log_file = max(log_files, key=lambda f: int(f.split('_')[1].split('.')[0]))
        turn_number = int(latest_log_file.split('_')[1].split('.')[0])

        log_path = os.path.join(logs_dir, latest_log_file)
        capture_path = os.path.join(captures_dir, f"turn_{turn_number:06d}.png")

        return turn_number, log_path, capture_path
    except (ValueError, FileNotFoundError):
        return -1, None, None

def read_turn_data(log_path):
    """Reads and parses the turn log JSON file."""
    if not log_path or not os.path.exists(log_path):
        return None
    with open(log_path, 'r') as f:
        return json.load(f)

def read_memory_files(memory_dir):
    """Reads all memory files and returns them as a dictionary."""
    memory_data = {}
    if not os.path.exists(memory_dir):
        return memory_data

    for filename in os.listdir(memory_dir):
        filepath = os.path.join(memory_dir, filename)
        if os.path.isfile(filepath):
            with open(filepath, 'r') as f:
                memory_data[filename] = f.read()
    return memory_data

def update_and_broadcast_data():
    """Checks for new data and broadcasts it to clients if found."""
    global latest_turn_number, latest_turn_data

    # Use absolute paths based on the script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(script_dir, '..', '..'))
    logs_dir = os.path.join(root_dir, 'logs')
    captures_dir = os.path.join(root_dir, 'captures')
    memory_dir = os.path.join(root_dir, 'memory')

    turn_number, log_path, capture_path = get_latest_turn_files(logs_dir, captures_dir)

    if turn_number > latest_turn_number:
        print(f"New turn detected: {turn_number}")
        latest_turn_number = turn_number

        turn_data = read_turn_data(log_path)
        memory_data = read_memory_files(memory_dir)

        # The capture path needs to be accessible by the browser, so we'll serve it
        # via an API endpoint and send the URL.
        capture_url = f"/api/screenshot/{turn_number}" if capture_path and os.path.exists(capture_path) else None

        latest_turn_data = {
            'turn_number': turn_number,
            'log': turn_data,
            'screenshot_url': capture_url,
            'memory': memory_data,
        }

        socketio.emit('update', latest_turn_data)

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    if path != "" and os.path.exists(app.static_folder + '/' + path):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')

@app.route('/api/latest')
def get_latest_data():
    """API endpoint to get the latest turn data for initial load."""
    if latest_turn_data is None:
        update_and_broadcast_data() # Initial check

    if latest_turn_data:
        return jsonify(latest_turn_data)
    return jsonify({'error': 'No data available yet.'}), 404

@app.route('/api/screenshot/<int:turn_number>')
def get_screenshot(turn_number):
    """Serves the screenshot for a specific turn."""
    from flask import send_from_directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(script_dir, '..', '..'))
    captures_dir = os.path.join(root_dir, 'captures')
    filename = f"turn_{turn_number:06d}.png"
    return send_from_directory(captures_dir, filename)

@socketio.on('connect')
def handle_connect():
    print('Client connected')
    # Send the latest data immediately on connection
    if latest_turn_data:
        emit('update', latest_turn_data)

class ChangeHandler(FileSystemEventHandler):
    def __init__(self, callback):
        self.callback = callback

    def on_created(self, event):
        self.callback()

    def on_modified(self, event):
        self.callback()

def start_monitoring():
    """Starts a watchdog observer to monitor file changes."""
    # Use absolute paths based on the script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(script_dir, '..', '..'))
    logs_dir = os.path.join(root_dir, 'logs')
    memory_dir = os.path.join(root_dir, 'memory')

    # Ensure directories exist before monitoring
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(memory_dir, exist_ok=True)

    event_handler = ChangeHandler(update_and_broadcast_data)
    observer = Observer()
    observer.schedule(event_handler, logs_dir, recursive=False)
    observer.schedule(event_handler, memory_dir, recursive=True)
    observer.start()
    print("Started monitoring logs and memory directories...")
    return observer

def run_app(debug=False):
    """Starts the web server and file monitoring."""
    observer = start_monitoring()
    try:
        update_and_broadcast_data()
        # When running from main.py, debug should be False.
        socketio.run(app, host='0.0.0.0', port=5000, debug=debug, allow_unsafe_werkzeug=debug)
    finally:
        observer.stop()
        observer.join()

if __name__ == '__main__':
    # For standalone testing of the web server.
    run_app(debug=True)
