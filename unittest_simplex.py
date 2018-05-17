import unittest
import numpy

from simplex import Simplex


class TestSimplexKeyGenerators(unittest.TestCase):
	def test_simplex_key_generators(self):
		simplex = Simplex(4, 20)

		simplex_keys_1 = []
		for simplex_key in simplex.simplex_keys():
			simplex_keys_1.append(simplex_key)

		simplex_keys_2 = []
		for simplex_key in simplex.simplex_keys_lazy_but_correct():
			simplex_keys_2.append(simplex_key)

		self.assertEqual(len(simplex_keys_1), len(simplex_keys_2))
		self.assertEqual(len(simplex_keys_1), 10626)

		for i in range(len(simplex_keys_1)):
			numpy.testing.assert_array_equal(simplex_keys_1[i], simplex_keys_2[i])


if __name__ == "__main__":
	unittest.main()
