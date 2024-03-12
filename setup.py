from setuptools import setup, find_packages

setup(
    name='src',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        # Any dependencies your package needs to work,
        # e.g., 'requests>=2.20.0'
    ],
    # Additional metadata about your package
    author='Your Name',
    author_email='your.email@example.com',
    description='A short description of your package',
    license='MIT',
    keywords='example project',
    url='http://example.com/MyPackage',  # Project home page
)
