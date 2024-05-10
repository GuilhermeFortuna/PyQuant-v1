from setuptools import setup, find_packages

setup(
    name='Backtester',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'pandas==2.2.2',
        'pandas_ta==0.3.14b0',
        'numpy==1.26.4',
        'dash==2.17.0',
        'dash-bootstrap-components==1.6.0',
        'dash-bootstrap-templates==1.1.2',
        'dash-core-components==2.0.0',
        'dash-renderer==1.9.1',
        'dash-table==5.0.0',
        'optuna-dashboard==0.15.1',
        'optuna==3.6.0',
    ]
)
