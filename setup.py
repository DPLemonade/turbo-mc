from setuptools import setup, find_packages


# Import version
__builtins__.__TURBO_MC_SETUP__ = True
from turbo_mc import __version__ as version


setup(
    name='turbo_mc',
    version=version,
    packages=find_packages(),
    install_requires=[
        'coverage',
        'cython',
        'scikit-surprise @ git+ssh://git@github.com/sprillo/Surprise.git@deeeaf889f6e24643753cd935fbae0f612771b6e#egg=scikit-surprise',
        'matplotlib',
        'numpy',
        'pandas',
        'pytest',
        'pytest-cov',
        'seaborn',
        'scipy',
        'sklearn'],
)
