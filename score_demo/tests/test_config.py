import unittest

from app.core.config import settings


class ConfigTests(unittest.TestCase):
    def test_settings_load_environment_only_from_local_env_file(self):
        self.assertEqual(settings.model_config.get("env_file"), ".env")


if __name__ == "__main__":
    unittest.main()
