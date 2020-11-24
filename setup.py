import sys

from setuptools import setup

for arg in sys.argv:
    if arg == "upload" or arg == "register":
        print("Package not intended for publication at this time.")
        sys.exit(-1)


setup(name="roerld",
      version="0.0.1",
      packages=["roerld"],
      install_requires=[
          "tensorflow>=2.2.0",
          "numpy>=1.19.0",
          "ray>=1.0",
          "gym>=0.17.3"
      ])
