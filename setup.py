from setuptools import setup, find_packages

setup(
    name='grouped_timeserie_cv',
    version='0.1',
    packages=find_packages(),
    install_requires=["plotly", 'pandas', 'scikit-learn', 'matplotlib'],
)