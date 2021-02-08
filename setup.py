from setuptools import find_packages, setup

setup(
    name='src',
    packages=find_packages(),
    test_suite='src.tests.test_all.suite'
)