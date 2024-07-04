# Turbo-CF

## Overview
This repo is the source code for **Turbo-CF (accepted in SIGIR 2024)**

## Features
- **Training-Free**: Operates without the need for model training.
- **Polynomial Graph Filters**: Utilizes efficient polynomial filters, makes it possible to easily compute in parallelism via GPU.
- **High accuracy**: Achieves near SOTA with very fast computation
  
## Installation
Clone the repository and install the necessary dependencies:
```bash
git clone https://github.com/jindeok/Turbo-CF.git
cd Turbo-CF
pip install -r requirements.txt
```

## Running
To run the Turbo-CF algorithm, use the provided script (dataset:'gowalla', 'yelp', 'amazon':

```bash
python main.py --dataset [dataset_name] --filter [filter_type] --alpha [alpha] --power [power]
```
example:
```bash
python main.py --dataset gowalla --filter 1
```

## Optimal parameters
We provide optimal parameters for each dataset herein.

- Gowalla
alpha (a): 0.6, power (s): 0.7 filter: 1
- Yelp
alpha (a): 0.6, power (s): 1 filter: 2
- Amazon-book
alpha (a): 0.5, power (s): 1.4 filter: 1


## Citation
If this work was helpful for your project, please kindly cite this in your paper

[ref] Jin-Duk Park, Yong-Min Shin, and Won-Yong Shin. "Turbo-CF: Matrix Decomposition-Free Graph Filtering for Fast Recommendation." In SIGIR 2024.   

@article{park2024turbo,    
  title={Turbo-CF: Matrix Decomposition-Free Graph Filtering for Fast Recommendation},   
  author={Park, Jin-Duk and Shin, Yong-Min and Shin, Won-Yong},   
  journal={arXiv preprint arXiv:2404.14243},   
  year={2024}
}
