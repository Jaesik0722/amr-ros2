from glob import glob

from setuptools import find_packages, setup

package_name = 'amr_docking'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
        ('share/' + package_name + '/markers', glob('markers/*.png')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Jaesik0722',
    maintainer_email='jslvjh@gmail.com',
    description='ArUco-based docking to the charging station and to carts.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'aruco_detector = amr_docking.aruco_detector:main',
            'dock_charger = amr_docking.commands:dock_charger',
            'undock_charger = amr_docking.commands:undock_charger',
            'dock_cart = amr_docking.commands:dock_cart',
            'release_cart = amr_docking.commands:release_cart',
            'cart_towing = amr_docking.cart_towing:main',
        ],
    },
)
