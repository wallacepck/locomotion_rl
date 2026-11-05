from setuptools import find_packages, setup

package_name = 'locomotion_rl'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='wallacepck',
    maintainer_email='142145699+wallacepck@users.noreply.github.com',
    description='Mujoco Playground simulation setup for a custom robot',
    license='Apache-2.0',
    extras_require={
    },
    entry_points={
        'console_scripts': [
        ],
    },
)
