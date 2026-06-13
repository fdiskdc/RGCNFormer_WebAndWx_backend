"""
Tests for API routes - Verify Flask app routes are registered.

Tests that all expected API endpoints exist and the health endpoint responds.
Note: Importing server.py triggers model loading; we mock heavy dependencies.
"""
import sys
import os
import json
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def _get_app():
    """Import and return the Flask app with mocked dependencies."""
    # Mock heavy dependencies before importing server
    mock_redis = MagicMock()
    mock_redis.ping.return_value = True
    mock_redis.get.return_value = None
    mock_redis.setex.return_value = True

    with patch('redis.Redis', return_value=mock_redis), \
         patch('torch.load', return_value={'model_state_dict': {}}), \
         patch('main_model.RNA_ClassQuery_Model') as MockModel:
        mock_model_instance = MagicMock()
        MockModel.return_value = mock_model_instance
        mock_model_instance.eval.return_value = mock_model_instance
        mock_model_instance.to.return_value = mock_model_instance

        import server
        return server.app


class TestAPIRoutes:
    """Test that all API routes are registered."""

    def test_health_route_registered(self):
        app = _get_app()
        rules = {rule.rule for rule in app.url_map.iter_rules()}
        assert '/api/health' in rules

    def test_wx_login_route_registered(self):
        app = _get_app()
        rules = {rule.rule for rule in app.url_map.iter_rules()}
        assert '/api/v1/wx/login' in rules

    def test_wx_submit_task_route_registered(self):
        app = _get_app()
        rules = {rule.rule for rule in app.url_map.iter_rules()}
        assert '/api/v1/wx-submit-task' in rules

    def test_wx_task_progress_route_registered(self):
        app = _get_app()
        rules = {rule.rule for rule in app.url_map.iter_rules()}
        assert '/api/v1/wx-task-progress/<job_id>' in rules

    def test_submit_task_route_registered(self):
        app = _get_app()
        rules = {rule.rule for rule in app.url_map.iter_rules()}
        assert '/api/v1/submit-task' in rules

    def test_results_route_registered(self):
        app = _get_app()
        rules = {rule.rule for rule in app.url_map.iter_rules()}
        assert '/api/v1/results/<job_id>' in rules

    def test_model_architecture_route_registered(self):
        app = _get_app()
        rules = {rule.rule for rule in app.url_map.iter_rules()}
        assert '/api/v1/model-architecture' in rules

    def test_model_graph_route_registered(self):
        app = _get_app()
        rules = {rule.rule for rule in app.url_map.iter_rules()}
        assert '/api/v1/model-graph' in rules

    def test_integrated_gradients_route_registered(self):
        app = _get_app()
        rules = {rule.rule for rule in app.url_map.iter_rules()}
        assert '/api/v1/integrated-gradients' in rules

    def test_visualize_gcn_aggregation_route_registered(self):
        app = _get_app()
        rules = {rule.rule for rule in app.url_map.iter_rules()}
        assert '/api/v1/visualize-gcn-aggregation' in rules

    def test_total_api_routes_count(self):
        """Verify we have at least 10 API routes (excluding static and HEAD/OPTIONS)."""
        app = _get_app()
        api_rules = {
            rule.rule for rule in app.url_map.iter_rules()
            if rule.rule.startswith('/api/')
        }
        assert len(api_rules) >= 10, f"Expected >= 10 API routes, got {len(api_rules)}: {api_rules}"


class TestHealthEndpoint:
    """Test the health check endpoint."""

    def test_health_returns_200(self):
        app = _get_app()
        with app.test_client() as client:
            response = client.get('/api/health')
            assert response.status_code == 200

    def test_health_returns_json(self):
        app = _get_app()
        with app.test_client() as client:
            response = client.get('/api/health')
            data = json.loads(response.data)
            assert 'status' in data
            assert data['status'] == 'ok'
