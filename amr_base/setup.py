from glob import glob

from setuptools import find_packages, setup

package_name = 'amr_base'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Jaesik0722',
    maintainer_email='jslvjh@gmail.com',
    description='Python driver for the MD Robot differential drive motor controller.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'base_node = amr_base.base_node:main',
        ],
    },
)
