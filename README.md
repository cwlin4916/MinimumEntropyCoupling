# Learning Minimum Entropy Coupling

## Description
Within an information-theoretic framework of steganography, the security of a communication procedure is determined by its correspondence to a coupling. Specifically, a minimum entropy coupling ensures maximal efficiency in secure procedures. The goal of the current project is to explore the use of deep learning methods to learn minimum entropy coupling. Additionally, this study examines the potential for improved heuristics in MECs, addressing the open question of whether faster and more efficient methods can be developed for secure data transmission. 

Following the approach used by Charton, who has done a series of research on “mathematical algorithm” learning, a transformer-based model is configured to learn the relationships between pairs of distributions, leveraging a supervised learning framework. The model architecture is based on the encoder-decoder structure with shared layers.


## Table of Contents

- [Description](#Description)
- [Src file](#SrcFileStructure)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [Testing](#testing)
- [License](#license)
- [Acknowledgments](#acknowledgments)
- [Directores](#directories)

## Src File Structure 

- `data_generation_algorithm1.py`: Contains the algorithm for generating data matrices with a specific entropy coupling characteristic.
  
- `model.py`: Defines the neural network architecture used for the entropy coupling model.
  
- `trainer.py`: Contains the Trainer class responsible for orchestrating the model training process with the provided data.
  
- `trainer1.py`: May represent an alternative or updated training script with variations from `trainer.py`.


## Installation

Instructions on how to run code. 
To train the model 
```bash
python trainer.py
```
To execute the data generation algorithm:

```bash
python data_generation_algorithm1.py
``` 
