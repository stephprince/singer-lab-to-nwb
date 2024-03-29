import setuptools

setuptools.setup(name="singer_lab_to_nwb",
                 description="singer_lab_to_nwb: research code for converting singer lab data to nwb format",
                 long_description=open('README.md').read(),
                 version="0.1.0",
                 url="https://github.com/stephprince/singer-lab-to-nwb",
                 license="MIT",
                 author="Steph Prince",
                 author_email="stephanie.m.prince1@gmail.com",
                 platforms="OS Independent",
                 classifiers=["Development Status :: 3 - Alpha",
                              "Intended Audience :: Science/Research",
                              "License :: OSI Approved :: MIT License",
                              "Operating System :: OS Independent",
                              "Programming Language :: Python",
                              "Topic :: Scientific/Engineering"],
                 packages=setuptools.find_packages(),
                 python_requires=">= 3.7",
                 )