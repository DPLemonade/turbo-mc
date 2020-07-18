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
        'surprise',
        'matplotlib',
        'numpy',
        'pandas',
        'pytest',
        'pytest-cov',
        'seaborn',
        'scipy',
        'sklearn'],
    dependency_links=[
        'git://github.com/sprillo/Surprise.git@243f45d815a965a4a10251172cd62c11387ac35f#egg=surprise'
    ],
)
