from setuptools import find_packages, setup

setup(
    name='wave_ensemble',
    packages=find_packages(),
    version='0.1.0',
    description='package to predict significant wave height using buoy historical observations with multiple ml alogrithms',
    author='felipe c. minuzzi',
    license='MIT',
    entry_points = '''
    [console_scripts]
    wave_ensemble=_cmd:cli
    ''',
)
