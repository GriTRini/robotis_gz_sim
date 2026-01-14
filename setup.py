import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'robotis_gz_sim'

setup(
    name=package_name,
    version='0.0.0',
    # 1. 패키지 탐색 (dsrpy 폴더에 __init__.py가 있어야 함)
    packages=find_packages(exclude=['test']),
    
    # 2. ★ 핵심 추가: 패키지 내부의 비-파이썬 파일(.so) 포함 설정 ★
    package_data={
        'robotis_gz_sim': ['dsrpy/*.so'],  # robotis_gz_sim/dsrpy 폴더 안의 모든 .so 파일을 포함
    },

    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        
        # Launch 파일 설치
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
        
        # Config/Rviz 파일 설치
        (os.path.join('share', package_name, 'config'), glob(os.path.join('config', '*'))),
        (os.path.join('share', package_name, 'rviz'), glob(os.path.join('rviz', '*'))),

        # 모델 파일 설치 (URDF & Meshes)
        (os.path.join('share', package_name, 'urdf/omy_f3m'), glob('urdf/omy_f3m/*')),
        (os.path.join('share', package_name, 'meshes/omy_f3m'), glob('meshes/omy_f3m/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='uon',
    maintainer_email='cd2918@naver.com',
    description='ROS 2 control and simulation package for robotis_gz_sim',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'trajectory = robotis_gz_sim.trajectory_bridge:main',
        ],
    },
)