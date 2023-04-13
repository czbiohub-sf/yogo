import unittest
import tempfile

from pathlib import Path
from contextlib import contextmanager

from yogo.data.dataloader import load_dataset_description

from .dataset_descs import ok_dataset_desc


@contextmanager
def generate_temp_dataset_desc(desc: str):
    t = tempfile.NamedTemporaryFile(mode='w', dir="/tmp", delete=False)
    t.write(desc)
    try:
        yield t.name
    finally:
        Path(t.name).unlink()



class TestDatasetDefinition(unittest.TestCase):
    def test_load_dataset_description(self):
        with generate_temp_dataset_desc(ok_dataset_desc) as fname:
            print(fname)
            load_dataset_description(fname)
