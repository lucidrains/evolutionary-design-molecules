from setuptools import setup, find_packages

setup(
  name = 'evolutionary-design-molecules',
  packages = find_packages(exclude=[]),
  version = '0.0.1',
  license='MIT',
  description = 'Evolutionary Design of Molecules',
  author = 'Phil Wang',
  author_email = 'lucidrains@gmail.com',
  long_description_content_type = 'text/markdown',
  url = 'https://github.com/lucidrains/evolutionary-design-molecules',
  keywords = [
    'artificial intelligence',
    'deep learning',
    'evolutionary algorithms'
  ],
  install_requires=[
    'beartype',
    'einops>=0.7.0',
    'torch>=2.0',
    'vector-quantize-pytorch>=1.12.1'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
