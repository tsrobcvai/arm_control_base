from setuptools import setup, find_packages

setup(
    name="arm_control_base",
    version="0.0.1",
    packages=["arm_control_base"] + ["arm_control_base." + pkg for pkg in find_packages(where=".")],
    package_dir={"arm_control_base": "."},
    install_requires=[
        "scipy",
        "pyserial>=3.4",
        "matplotlib",
        "pyyaml",
        "redis",
        "easydict",
        "hydra-core",
        "docutils",
    ],
)

