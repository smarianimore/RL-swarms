from setuptools import setup, find_packages

#setup(
    #name="slime_environments",
    #version="1.0.0",
    #packages=find_packages(),
    #install_requires=['gymnasium', 'pygame']
#)
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
