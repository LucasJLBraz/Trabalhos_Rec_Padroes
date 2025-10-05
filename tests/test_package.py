# -*- coding: utf-8 -*-
"""
Unit tests for the trabalhos_rec_padroes package.
"""

import sys
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).resolve().parents[1] / 'src'
sys.path.insert(0, str(src_path))

import unittest
from trabalhos_rec_padroes import __version__


class TestPackage(unittest.TestCase):
    """Test the package metadata."""

    def test_version(self):
        """Test that the package version is defined."""
        self.assertIsNotNone(__version__)
        self.assertEqual(__version__, '0.1.0')


if __name__ == '__main__':
    unittest.main()
