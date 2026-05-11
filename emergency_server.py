from flask import Flask

app = Flask(name)

emergency_status = "SAFE"
location_link = "Not Available"


@app.route('/')
def home():
    return f"""
    <h1>Drive Guardian Emergency Monitor</h1>
    <h2>Status: {emergency_status}</h2>
    <h3>Location:</h3>

    <a href="{location_link}" target="_blank">
        Open Live Location
    </a>
    """


def update_emergency(status, location):
    global emergency_status
    global location_link

    emergency_status = status
    location_link = location


if name == 'main':
    app.run(host='0.0.0.0', port=5000)