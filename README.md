# Learning Minimum Entropy Coupling

## Description
Within an information-theoretic framework of [steganography](https://arxiv.org/abs/2210.14889), the security of a communication procedure is determined by its correspondence to a coupling. Specifically, a minimum entropy coupling ensures maximal efficiency in secure procedures. The goal of the current project is to explore the use of deep learning methods to learn minimum entropy coupling. Additionally, this study examines the potential for improved heuristics in MECs, addressing the open question of whether faster and more efficient methods can be developed for secure data transmission. Logging of experiments is [here](https://www.overleaf.com/read/xtfyrzgpnwbn#657863)

## Table of Contents

- [Description](#Description)
- [Src file](#SrcFileStructure)

## Src File Structure 

- `data_mec_gen.py`: Contains the algorithm for generating mec data coupling. 
  
- `mec_trainer.py`: Defines the the boiler plate code for training mec model.
