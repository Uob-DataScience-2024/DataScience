import unittest
from loguru import logger
from rich.logging import RichHandler

logger.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])


class TotalTest(unittest.TestCase):
    def test_project_init(self):
        self.assertEqual(True, True)


if __name__ == '__main__':
    unittest.main()
