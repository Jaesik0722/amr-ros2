from glob import glob

from setuptools import setup

package_name = 'amr_description'

setup(
    name=package_name,
    version='0.1.0',
    packages=[],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/urdf', glob('urdf/*.xacro')),
        ('share/' + package_name + '/rviz', glob('rviz/*.rviz')),
        ('share/' + package_name + '/meshes', glob('meshes/*.stl') + glob('meshes/*.dae')),
    ],
    install_requires=['setuptools'],
    tests_require=['pytest'],
    zip_safe=True,
    maintainer='Jaesik0722',
    maintainer_email='jslvjh@gmail.com',
    description='URDF model of the towing AMR.',
    license='Apache-2.0',
)
