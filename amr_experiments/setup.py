from glob import glob

from setuptools import find_packages, setup

package_name = 'amr_experiments'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/maps', glob('maps/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Jaesik0722',
    maintainer_email='jslvjh@gmail.com',
    description='Repeated-trial experiments (docking, engagement, towing) with CSV logs.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'analyze = amr_experiments.analyze:main',
            'compare = amr_experiments.compare:main',
            'e1_charger_docking = amr_experiments.e1_charger_docking:main',
            'e2_cart_docking = amr_experiments.e2_cart_docking:main',
            'e3_engagement = amr_experiments.e3_engagement:main',
            'e4_corridor = amr_experiments.e4_corridor:main',
            'e5_towing = amr_experiments.e5_towing:main',
            'e6_endurance = amr_experiments.e6_endurance:main',
            'e7_scenario = amr_experiments.e7_scenario:main',
        ],
    },
)
