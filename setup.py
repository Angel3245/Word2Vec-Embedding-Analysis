from setuptools import setup

setup(
   name='Neural models for word vector representation',
   version='1.0',
   description='Train and evaluate neural models (CBOW and skipgram) for word vector representation',
   author='Jose Ángel Pérez Garrido',
   author_email='jpgarrido19@esei.uvigo.es',
   packages=['word2vec'],
   install_requires=['wheel', 'bar', 'greek'], #external packages as dependencies
)