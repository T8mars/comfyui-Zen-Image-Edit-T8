import struct
import tempfile
import unittest
from pathlib import Path

from checkpoint import metadata


class MetadataTest(unittest.TestCase):
    def test_rejects_short_header_truncated_header_and_invalid_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "broken.safetensors"
            null_metadata = b'{"__metadata__":null}'
            for contents in (
                b"short",
                struct.pack("<Q", 32) + b"{}",
                struct.pack("<Q", len(null_metadata)) + null_metadata,
            ):
                path.write_bytes(contents)
                with self.assertRaises(ValueError):
                    metadata(path)


if __name__ == "__main__":
    unittest.main()
