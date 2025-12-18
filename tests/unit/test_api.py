from fastapi.testclient import TestClient

from precise_mrd.api import app

client = TestClient(app)


def test_health_check():
    """Test the /health API endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["api_status"] == "ok"
    assert "db_status" in data
    assert "cache_status" in data


