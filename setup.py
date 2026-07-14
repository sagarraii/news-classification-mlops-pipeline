from setuptools import find_packages, setup
from typing import List

HYPHEN_E_DOT = "-e ."


def get_requirements(file_path: str) -> List[str]:
    """
    Returns a list of requirements from requirements.txt
    """

    requirements = []

    with open(file_path) as file_obj:
        requirements = [
            req.strip()
            for req in file_obj.readlines()
            if req.strip()
        ]

    if HYPHEN_E_DOT in requirements:
        requirements.remove(HYPHEN_E_DOT)

    return requirements


setup(
    name="NLP_News_Classification",
    version="0.0.1",
    author="Sagar Rai",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt") # make sure to use requirements-dev.txt
)

