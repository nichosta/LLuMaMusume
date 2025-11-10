from flask import Flask, jsonify, send_from_directory
from flask_socketio import SocketIO, emit
import os
import time
import json
import threading

# Construct the absolute path for the static folder
script_dir = os.path.dirname(os.path.abspath(__file__))
static_folder_path = os.path.abspath(os.path.join(script_dir, '..', 'frontend', 'build'))

app = Flask(__name__, static_folder=static_folder_path, static_url_path='/')
socketio = SocketIO(app, cors_allowed_origins="*")

# In-memory state
latest_turn_data = None
latest_turn_number = -1
last_mod_times = {}

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
    try:
        with open(log_path, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError:
        print(f"Error decoding JSON from {log_path}")
        return None

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
    with app.app_context():
        global latest_turn_number, latest_turn_data

        script_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.abspath(os.path.join(script_dir, '..', '..'))
        logs_dir = os.path.join(root_dir, 'logs')
        captures_dir = os.path.join(root_dir, 'captures')
        memory_dir = os.path.join(root_dir, 'memory')

        turn_number, log_path, capture_path = get_latest_turn_files(logs_dir, captures_dir)

        if turn_number != -1 and turn_number > latest_turn_number:
            latest_turn_number = turn_number
            print(f"New turn detected: {turn_number}")

        print("Broadcasting update...")
        turn_data = read_turn_data(log_path)
        memory_data = read_memory_files(memory_dir)
        capture_url = f"/api/screenshot/{latest_turn_number}" if capture_path and os.path.exists(capture_path) else None

        latest_turn_data = {
            'turn_number': latest_turn_number,
            'log': turn_data,
            'screenshot_url': capture_url,
            'memory': memory_data,
        }

        socketio.emit('update', latest_turn_data)

def poll_for_changes():
    """Polls the logs and memory directories for changes."""
    global last_mod_times
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(script_dir, '..', '..'))
    logs_dir = os.path.join(root_dir, 'logs')
    memory_dir = os.path.join(root_dir, 'memory')

    monitored_dirs = [logs_dir, memory_dir]

    for d in monitored_dirs:
        os.makedirs(d, exist_ok=True)
        if d not in last_mod_times:
            last_mod_times[d] = 0

    while True:
        changed = False
        for d in monitored_dirs:
            try:
                # Check for latest modification time in the directory
                files = os.listdir(d)
                if not files:
                    continue
                latest_mod_time = max(os.path.getmtime(os.path.join(d, f)) for f in files)
                if latest_mod_time > last_mod_times.get(d, 0):
                    last_mod_times[d] = latest_mod_time
                    changed = True
            except FileNotFoundError:
                # A file might be deleted between listdir and getmtime
                continue

        if changed:
            update_and_broadcast_data()

        time.sleep(1)

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
        update_and_broadcast_data()

    if latest_turn_data:
        return jsonify(latest_turn_data)
    return jsonify({'error': 'No data available yet.'}), 404

@app.route('/api/screenshot/<int:turn_number>')
def get_screenshot(turn_number):
    """Serves the screenshot for a specific turn."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(script_dir, '..', '..'))
    captures_dir = os.path.join(root_dir, 'captures')
    filename = f"turn_{turn_number:06d}.png"
    return send_from_directory(captures_dir, filename)

@socketio.on('connect')
def handle_connect():
    print('Client connected')
    if latest_turn_data:
        emit('update', latest_turn_data)

def run_app(debug=False):
    """Starts the web server and file monitoring."""
    update_and_broadcast_data()
    socketio.start_background_task(poll_for_changes)
    socketio.run(app, host='0.0.0.0', port=5000, debug=debug, allow_unsafe_werkzeug=debug, use_reloader=False)

if __name__ == '__main__':
    run_app(debug=True)
