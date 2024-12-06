from setuptools import setup, find_packages

setup(
    name="rl_swarms",  # Nome del pacchetto principale
    version="0.0.1",  # Versione
    packages=find_packages(include=["slime_environments", "refactoring", "slime_environments.*", "refactoring.*"]),
    python_requires=">=3.7",
    install_requires=[
        "gymnasium",
        "pygame"
    ]
)
