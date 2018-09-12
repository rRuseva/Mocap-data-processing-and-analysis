# import pytest
import unittest
import tools.py as t


def test_marker_index():
    assert t.marker_index(["ROWR", "RIWR", "LOWR", "LIWR"], "LIWR") == 3


if __name__ == '__main__':
    unittest.main()
