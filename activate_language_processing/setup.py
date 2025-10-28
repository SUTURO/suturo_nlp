## ! DO NOT MANUALLY INVOKE THIS setup.py, USE CATKIN INSTEAD

from setuptools import setup, find_packages
from glob import glob
import os

package_name = 'activate_language_processing'

# collect scripts (exclude __init__.py and hidden files)
script_files = [
    s for s in glob(os.path.join('scripts', '*.py'))
    if not os.path.basename(s).startswith('.') and os.path.basename(s) != '__init__.py'
]

data_files = [
    (os.path.join('share', package_name), ['package.xml']),
]

# include launch files if present
launch_files = glob(os.path.join('launch', '*'))
if launch_files:
    data_files.append((os.path.join('share', package_name, 'launch'), launch_files))

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['scripts', 'tests']),
    package_dir={'': '.'},
    scripts=script_files,
    data_files=data_files,
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='maintainer',
    maintainer_email='maintainer@example.com',
    description='Language processing package',
    license='Apache-2.0',
    tests_require=['pytest'],
)

