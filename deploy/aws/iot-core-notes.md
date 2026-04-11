# AWS IoT Core — Production Ingestion Path

In production, replace the local Mosquitto broker with **AWS IoT Core** as the MQTT endpoint.

## Architecture

```
BMW i4 Edge Device / Simulator
        │
        │  MQTT over TLS (port 8883)
        ▼
  AWS IoT Core
  Topic: uvtwin/telemetry/{vehicle_id}
        │
        │  IoT Rule → Lambda (optional transform)
        │  OR direct MQTT bridge to ECS backend
        ▼
  ECS Fargate (FastAPI)
  INGESTION_MODE=mqtt
  MQTT_HOST=<iot-endpoint>.iot.<region>.amazonaws.com
  MQTT_PORT=8883
```

## Setup steps

1. Create an IoT Thing: `aws iot create-thing --thing-name bmw-i4-twin`
2. Create and attach a certificate + policy allowing `iot:Publish` on `uvtwin/telemetry/*`
3. Download the device certificate, private key, and Amazon Root CA
4. Set env vars on the ECS task:
   - `MQTT_HOST` = `<endpoint>.iot.<region>.amazonaws.com`
   - `MQTT_PORT` = `8883`
   - `MQTT_TLS_CA` = path to AmazonRootCA1.pem
   - `MQTT_TLS_CERT` = path to device cert
   - `MQTT_TLS_KEY` = path to private key
5. Update `MqttConsumer.start()` to pass TLS context to paho-mqtt

## IoT Rule (optional — for fan-out to other AWS services)

```sql
SELECT * FROM 'uvtwin/telemetry/#'
```

Actions:
- → Kinesis Data Streams (for analytics)
- → S3 (for raw telemetry archival)
- → Lambda (for alerting / SNS notifications)
