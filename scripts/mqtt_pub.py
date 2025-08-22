#!/usr/bin/env python3
import paho.mqtt.client as mqtt
import json, time, os

BROKER = "broker.hivemq.com"
PORT = 1883
TOPIC = f"uvtwin/telemetry/{os.getenv('USER') or os.getenv('USERNAME') or 'Anadi'}"

c = mqtt.Client()
c.connect(BROKER, PORT, 60)

# Send 5 telemetry messages
for i in range(5):
    msg = {"speed": 50+i*5, "rpm": 2000+i*100, "temp": 80+i}
    c.publish(TOPIC, json.dumps(msg))
    print("Published:", msg)
    time.sleep(2)

c.disconnect()
