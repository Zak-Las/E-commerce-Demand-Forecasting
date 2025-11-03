from fastapi.testclient import TestClient
from src.service.app import app

client = TestClient(app)

def test_health():
    r = client.get('/health')
    assert r.status_code == 200
    assert r.json()['status'] == 'ok'

def test_forecast_dummy():
    payload = {"product_ids": ["ITEM_1", "ITEM_2"], "horizon": 5}
    r = client.post('/forecast', json=payload)
    assert r.status_code == 200
    data = r.json()
    assert 'forecasts' in data
    assert len(data['forecasts']) == 2 * 5
