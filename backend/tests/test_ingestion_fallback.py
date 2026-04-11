from app.ingestion.ingestion_runner import IngestionRunner


def test_set_mode_switches_to_csv() -> None:
    runner = IngestionRunner()

    runner._set_mode("csv")

    assert runner.mode == "csv"


class _FailingMqttConsumer:
    def start(self) -> None:
        raise RuntimeError("mqtt unavailable")


class _SuccessfulMqttConsumer:
    def __init__(self) -> None:
        self.called = False

    def start(self) -> None:
        self.called = True


class _CsvFallbackSpy:
    def __init__(self) -> None:
        self.called = False

    def start(self) -> None:
        self.called = True


def test_start_switches_to_csv_when_mqtt_fails() -> None:
    csv_fallback = _CsvFallbackSpy()
    runner = IngestionRunner(
        mqtt_consumer=_FailingMqttConsumer(),
        csv_fallback=csv_fallback,
    )

    runner.start()

    assert runner.mode == "csv"
    assert csv_fallback.called is True


def test_start_keeps_mqtt_mode_when_mqtt_succeeds() -> None:
    mqtt_consumer = _SuccessfulMqttConsumer()
    csv_fallback = _CsvFallbackSpy()
    runner = IngestionRunner(
        mqtt_consumer=mqtt_consumer,
        csv_fallback=csv_fallback,
    )

    runner.start()

    assert runner.mode == "mqtt"
    assert mqtt_consumer.called is True
    assert csv_fallback.called is False
