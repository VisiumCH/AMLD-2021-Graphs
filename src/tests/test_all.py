import unittest

from src.tests import test_data
from src.tests import test_models
from src.tests import test_visualization
from src.tests import test_api

suites = []
suites.append(test_data.suite)
suites.append(test_models.suite)
suites.append(test_visualization.suite)
suites.append(test_api.suite)

suite = unittest.TestSuite(suites)

if __name__ == '__main__':  # pragma: no cover
    unittest.TextTestRunner(verbosity=2).run(suite)
