from glob import glob

from setuptools import find_packages, setup

package_name = 'amr_capture'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', glob('config/*')),
        ('share/' + package_name + '/launch', glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Jaesik0722',
    maintainer_email='jslvjh@gmail.com',
    description='Sensor-rig recording and dataset export for Gaussian Splatting.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'crop_operator = amr_capture.crop_operator:main',
            'export_dataset = amr_capture.export_dataset:main',
            'pcd_to_mesh = amr_capture.pcd_to_mesh:main',
        ],
    },
)
