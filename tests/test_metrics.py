from src.metrics import classification_report_dict


def test_metrics_runs():
    y = [0, 1, 1, 0]
    p = [0.1, 0.8, 0.7, 0.2]
    out = classification_report_dict(y, p)
    assert out['accuracy'] == 1.0
    assert 0 <= out['ece'] <= 1
