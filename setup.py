import os
from setuptools import setup, find_packages

# Change directory to allow installation from anywhere
script_folder = os.path.dirname(os.path.realpath(__file__))
os.chdir(script_folder)

with open("requirements-compiled.txt") as f:
    lines = f.read().splitlines()
requirements = [l for l in lines
                if l and not l.startswith(("#", "--"))]

setup(
    name="AutoVLA",
    version="1.0.0",
    author="Zewei Zhou, Tianhui Cai, Seth Z. Zhao, Yun Zhang, Zhiyu Huang, Bolei Zhou, Jiaqi Ma",
    description="AutoVLA: A Vision-Language-Action Model for End-to-End Autonomous Driving with Adaptive Reasoning and Reinforcement Fine-Tuning",
    url="https://github.com/ucla-mobility/AutoVLA",
    packages=find_packages(where=script_folder),
    package_dir={"": "."},
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    license="Academic-Software-License",
    install_requires=requirements,
)