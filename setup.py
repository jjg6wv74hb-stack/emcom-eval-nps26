from pathlib import Path

from setuptools import find_packages, setup


def read_requirements(path: str) -> list[str]:
    requirements = []
    for raw_line in Path(path).read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        requirements.append(line)
    return requirements


setup(
    name="pgg",
    version="1.0.1",
    packages=find_packages(),
    install_requires=read_requirements("requirements.txt"),
)
