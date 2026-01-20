from setuptools import find_packages,setup
from typing import List
def get_requirements(filepath:str)->List[str]:
    '''This will return list of requirments'''
    reqs = []
    with open(filepath, "r") as fp:
        for line in fp:
            line = line.strip()
            if line == '-e .':
                continue
            
            if line:  
                reqs.append(line)
    
    return reqs

setup(
    name='mlproject',
    version='0.0.1',
    author='Hanni',
    author_email='hannikanchap11@gmail.com',
    packages=find_packages(),
    install_requires = get_requirements('requirements.txt')
)