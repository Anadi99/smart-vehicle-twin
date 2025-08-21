#!/usr/bin/env python3
import os, json, pandas as pd, time
import paho.mqtt.client as mqtt

HOST = os.getenv("MQTT_HOST", "broker.hivemq.com")
PORT = int(os.getenv("MQTT_PORT", "1883"))
TOPIC = os.getenv("MQTT_TOPIC", f"uvtwin/telemetry/{os.getenv('USERNAME','user')}")
LATEST = "data/latest.csv"

df = pd.read_csv(LATEST)
msg = df.iloc[-1].to_dict()
msg["ts_published"] = time.time()

c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
c.connect(HOST, PORT, 60)
payload = json.dumps(msg)
c.publish(TOPIC, payload, qos=0, retain=False)
c.disconnect()
print(f"Published {len(payload)} bytes to mqtt://{HOST}:{PORT}/{TOPIC}")
