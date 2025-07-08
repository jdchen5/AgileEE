from setuptools import setup, find_packages

setup(
    name="agileee",
    version="0.1.0",
    description="AgileEE - Agile Engineering Excellence",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "streamlit",
        "pandas",
        "numpy",
        "scikit-learn",
        "pycaret",
        # Add other dependencies from your requirements.txt
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "agileee-app=agileee.main:main",
        ],
    },
)