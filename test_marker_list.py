import pytest
import unittest
import tools as t


# # returns the index of a given marker from given marker list
# def marker_index(mlist, mname):
#     for i, m in enumerate(mlist):
#         if(m == mname):
#             return i

#     return False


# # returns the marker name for the given index from given list of markers
# def marker_name(mlist, index):
#     if(index < len(mlist)):
#         return mlist[index]
#     else:
#         return False


def test_marker_index_found():
    assert t.marker_index(["ROWR", "RIWR", "LOWR", "LIWR"], "LIWR") == 3


def test_marker_index_notfound():
    assert t.marker_index(["ROWR", "RIWR", "LOWR", "LIWR"], "RFWT") == False


def test_marker_name_found():
    assert t.marker_name(["ROWR", "RIWR", "LOWR", "LIWR"], 2) == "LOWR"


def test_marker_name_notfound():
    assert t.marker_name(["ROWR", "RIWR", "LOWR", "LIWR"], 5) == False


if __name__ == '__main__':
    unittest.main()
