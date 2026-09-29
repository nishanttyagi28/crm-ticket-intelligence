from ticketintel.monitor import missing_text_rate, monitor_batch, total_variation


def test_total_variation_matches_and_disjoint():
    classes = ["a", "b"]
    assert total_variation({"a": 0.5, "b": 0.5}, {"a": 0.5, "b": 0.5}, classes) == 0.0
    assert total_variation({"a": 1.0, "b": 0.0}, {"a": 0.0, "b": 1.0}, classes) == 1.0


def test_monitor_flags_a_shifted_batch():
    report = monitor_batch(
        reference_labels=["card"] * 80 + ["loan"] * 20,
        batch_labels=["loan"] * 50,
        batch_texts=["fee"] * 49 + [""],
        classes=["card", "loan"],
    )
    assert report["rows"] == 50
    assert report["missing_text_rate"] == 0.02
    assert report["flag"] is True
    assert missing_text_rate(["", "ok"]) == 0.5
