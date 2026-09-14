from io import BytesIO
import os
from pathlib import Path
import sys
import tempfile
import unittest

from fastapi.testclient import TestClient
from PIL import Image


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
TEST_DATABASE_DIRECTORY = tempfile.TemporaryDirectory()
os.environ["FIELDCARE_DB_PATH"] = str(
    Path(TEST_DATABASE_DIRECTORY.name) / "fieldcare-test.db"
)

from app import MAX_UPLOAD_BYTES, app  # noqa: E402


def png_bytes() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (320, 320), color=(50, 140, 55)).save(buffer, format="PNG")
    return buffer.getvalue()


class PredictionApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.client.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client.__exit__(None, None, None)
        TEST_DATABASE_DIRECTORY.cleanup()

    def test_health_reports_loaded_model(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ready")
        self.assertEqual(response.json()["class_count"], 25)

    def test_predict_returns_model_result(self):
        response = self.client.post(
            "/predict",
            files={"file": ("leaf.png", png_bytes(), "image/png")},
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("class_key", payload)
        self.assertIsInstance(payload["history_id"], int)
        self.assertIn("crop", payload)
        self.assertIn("disease", payload)
        self.assertGreaterEqual(payload["confidence"], 0)
        self.assertLessEqual(payload["confidence"], 1)

    def test_successful_prediction_is_available_in_history(self):
        image = png_bytes()
        prediction = self.client.post(
            "/predict",
            files={"file": ("history-leaf.png", image, "image/png")},
        )
        self.assertEqual(prediction.status_code, 200)
        prediction_payload = prediction.json()

        history = self.client.get("/history", params={"q": "history-leaf"})
        self.assertEqual(history.status_code, 200)
        records = history.json()["records"]
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["id"], prediction_payload["history_id"])
        self.assertEqual(records[0]["class_key"], prediction_payload["class_key"])

        stored_image = self.client.get(records[0]["image_url"])
        self.assertEqual(stored_image.status_code, 200)
        self.assertEqual(stored_image.headers["content-type"], "image/png")
        self.assertEqual(stored_image.content, image)

        stats = self.client.get("/history/stats")
        self.assertEqual(stats.status_code, 200)
        self.assertGreaterEqual(stats.json()["total_scans"], 1)

    def test_history_rejects_unknown_category(self):
        response = self.client.get("/history", params={"category": "urgent"})
        self.assertEqual(response.status_code, 422)

    def test_predict_rejects_invalid_image(self):
        response = self.client.post(
            "/predict",
            files={"file": ("leaf.png", b"not an image", "image/png")},
        )
        self.assertEqual(response.status_code, 400)

    def test_predict_rejects_oversized_image(self):
        response = self.client.post(
            "/predict",
            files={"file": ("leaf.png", b"x" * (MAX_UPLOAD_BYTES + 1), "image/png")},
        )
        self.assertEqual(response.status_code, 413)

    def test_every_model_class_has_treatment_guidance(self):
        for class_info in app.state.classes:
            response = self.client.get(f"/treatments/{class_info['key']}")
            self.assertEqual(response.status_code, 200, class_info["key"])
            payload = response.json()
            self.assertEqual(payload["class_key"], class_info["key"])
            self.assertTrue(payload["cure_points"])
            self.assertTrue(payload["prevention_points"])

    def test_unknown_treatment_returns_not_found(self):
        response = self.client.get("/treatments/not-a-model-class")
        self.assertEqual(response.status_code, 404)


if __name__ == "__main__":
    unittest.main()
