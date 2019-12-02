from setuptools import setup, find_packages

setup(
    name='eegip',
    version='0.0.1',
    url='https://github.com/christian-oreilly/eegip.git',
    author="Christian O'Reilly",
    author_email='christian.oreilly@gmail.com',
    description='Code for connectivity analyse using MNE and the EEGIP database.',
    packages=find_packages(),    
    install_requires=['numpy >= 1.11.1', 'matplotlib >= 1.5.1', 'tqdm', 'pandas', 'scipy', 'mne', 'seaborn', 'xarray',
                      'trimesh', 'pyyaml', 'jinja2', 'xlrd', 'parse', 'scikit-image', 'parse_type', 'cortex', 'pybids'],
)
