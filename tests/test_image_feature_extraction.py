import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from scripts.extract_image_features import extract_features_with_reason


class ImageFeatureExtractionTests(unittest.TestCase):
    def test_extract_features_with_reason_reports_unreadable_image(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "invalid.png"
            path.write_bytes(b"not-a-real-image")

            features, reason = extract_features_with_reason(path)

            self.assertIsNone(features)
            self.assertIsNotNone(reason)
            self.assertIn("read", reason.lower())

    def test_extract_features_with_reason_rejects_images_without_a_clear_face(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "no_face.png"
            img = np.full((224, 224, 3), 255, dtype=np.uint8)
            cv2.imwrite(str(path), img)

            features, reason = extract_features_with_reason(path)

            self.assertIsNone(features)
            self.assertIsNotNone(reason)
            self.assertIn("face", reason.lower())


if __name__ == "__main__":
    unittest.main()
