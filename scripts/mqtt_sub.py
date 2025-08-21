#!/usr/bin/env python3
import os, json
import paho.mqtt.client as mqtt

HOST = os.getenv("MQTT_HOST", "broker.hivemq.com")
PORT = int(os.getenv("MQTT_PORT", "1883"))
TOPIC = os.getenv("MQTT_TOPIC", f"uvtwin/telemetry/{os.getenv('USERNAME','user')}")

def on_connect(client, userdata, flags, reason_code, properties=None):
    print("Connected:", reason_code)
    client.subscribe(TOPIC)

def on_message(client, userdata, msg):
    try:
        print("MSG:", json.loads(msg.payload.decode()))
    except Exception:
        print("RAW:", msg.payload)

c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
c.on_connect = on_connect
c.on_message = on_message
c.connect(HOST, PORT, 60)
print(f"Subscribing to mqtt://{HOST}:{PORT}/{TOPIC}")
c.loop_forever()
