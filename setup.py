from setuptools import setup

__version__ = '0.1.0'

with open('README.md') as f:
    long_description = f.read()

setup(
    name='HRED-VHRED',
    version=__version__,
    description='The original implementation of HRED and VHRED dialogue model '
                'by Serban and Sordoni. Modified by Cong Feng',
    url='https://github.com/cgsdfc/HRED-VHRED.git',
    author='Iulian Serban, Alessandro Sordoni, Cong Feng',
    author_email='julianserban@gmail.com, cgsdfc@126.com',
    keywords=[
        'hierarchical recurrent encoder decoder',
        'variational hierarchical recurrent encoder decoder',
        'natural dialogue generation',
        'natural language processing',
        'computational linguistics',
    ],
    scripts=[
        'bin/chat.py',
        'bin/compute_dialogue_embeddings.py',
        'bin/convert_text2dict.py',
        'bin/convert_wordemb_dict2emb_matrix.py',
        'bin/create_text_file_for_tests.py',
        'bin/evaluate.py',
        'bin/generate_encodings.py',
        'bin/sample.py',
        'bin/split_documents_by_dialogues.py',
        'bin/split_examples_by_token.py',
        'bin/train.py',
        'monitor/get_logfile.py',
    ],
    packages=[
        'serban',
        'serban.tests',
    ],
    package_data={
        'serban.tests': [
            'bleu/*',
            'data/*',
        ]
    },
    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved ::  GPL v3',
        'Programming Language :: Python :: 3',
        'Topic :: Text Processing :: Linguistic',
    ],
    license='LICENCE.txt',
    long_description=long_description,
    install_requires=[
        'scikit-learn',
        'pyenchant',
        'numpy',
        'theano',
    ],
)
