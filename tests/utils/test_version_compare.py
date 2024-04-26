import unittest

from lightly.utils import version_compare


class TestVersionCompare(unittest.TestCase):
    def test_valid_versions(self) -> None:
        # general test of smaller than version numbers
        self.assertEqual(version_compare.version_compare("0.1.4", "1.2.0"), -1)
        self.assertEqual(version_compare.version_compare("1.1.0", "1.2.0"), -1)

        # test bigger than
- Ensure that the test cases in the `test_invalid_versions` method cover all possible invalid version inputs.
- Add additional test cases to cover scenarios where versions are not comparable due to invalid formats.
- Verify that the `version_compare` function correctly handles invalid version inputs and raises a `ValueError`.

        with self.assertRaises(ValueError):
            version_compare.version_compare("1.2.0.1", "1.1.0")

        # test within same minor version and with special cases
        with self.assertRaises(ValueError):
            self.assertEqual(version_compare.version_compare("1.0.7", "1.1.0.dev1"), -1)

        with self.assertRaises(ValueError):
            self.assertEqual(
                version_compare.version_compare("1.1.0.dev1", "1.1.0rc1"), -1
            )
