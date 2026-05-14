from setuptools import setup, find_packages

setup(
    name="eve-dimming-index",
    version="2.0.0",
    description="EVE Dimming Index analysis tools",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas", 
        "scipy",
        "matplotlib",
        "astropy",
        "sunpy",
        "plotly",
        "scikit-learn",
        "scikit-image",
        "seaborn",
        "statsmodels",
        "networkx",
        "pillow",
        "h5py",
        "requests",
        "colorlover",
        "cufflinks",
        "peakutils",
    ],
    python_requires=">=3.9",
) 