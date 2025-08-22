#!/usr/bin/env python3
import paho.mqtt.client as mqtt
import json, os

HOST="broker.hivemq.com"
PORT=1883
TOPIC=f"uvtwin/telemetry/{os.getenv('USER') or os.getenv('USERNAME') or 'anon'}"
DATA_FILE="data/telemetry.jsonl"

os.makedirs("data", exist_ok=True)

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print(f"Connected: Success (subscribed to {TOPIC})")
        client.subscribe(TOPIC)
    else:
        print("Connection failed with code", rc)

def on_message(client, userdata, msg):
    try:
        payload = msg.payload.decode()
        j = json.loads(payload)
        print("MSG:", j)

        # Save to file
        with open(DATA_FILE, "a") as f:
            f.write(json.dumps(j) + "\n")

    except Exception as e:
        print("RAW:", msg.payload, "ERR:", e)

c = mqtt.Client()
c.on_connect=on_connect
c.on_message=on_message
c.connect(HOST, PORT, 60)
c.loop_forever()
