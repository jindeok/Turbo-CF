# Turbo-CF

## Overview and Features
This repo is the source code for **Turbo-CF (SIGIR 2024)**

<img src="figure1.jpg" alt="Turbo-CF Workflow" width="450">
 
- **Training-Free**: Operates without the need for model training.
- **Polynomial Graph Filters**: Utilizes efficient polynomial filters, makes it possible to easily compute in parallelism via GPU.
- **High Accuracy**: Achieves near SOTA with very fast computation
  
## Installation
Clone the repository and install the necessary dependencies:
```bash
git clone https://github.com/jindeok/Turbo-CF.git
cd Turbo-CF
pip install -r requirements.txt
```

## Running
To run the Turbo-CF (dataset:'gowalla', 'yelp', 'amazon'):

```bash
python main.py --dataset [dataset_name] --filter [filter_type] --alpha [alpha] --power [power]
```
example:
```bash
python main.py --dataset gowalla --filter 1
```

Can also be run in jupyter env: main.ipynb

## Optimal parameters
We provide the optimal parameters for each dataset herein.

- Gowalla
alpha (a): 0.6, power (s): 0.7 filter: 1
- Yelp
alpha (a): 0.6, power (s): 1 filter: 2
- Amazon-book
alpha (a): 0.5, power (s): 1.4 filter: 1

(We note that as Turbo-CF is a rule-based method, we use all available dataset (training + validation) for the graph filtering (inference).)

## Citation
If this work was helpful for your project, please kindly cite this in your paper

[ref] Jin-Duk Park, Yong-Min Shin, and Won-Yong Shin. "Turbo-CF: Matrix Decomposition-Free Graph Filtering for Fast Recommendation." In SIGIR 2024.   

@inproceedings{park2024turbo,  
  title={Turbo-cf: Matrix decomposition-free graph filtering for fast recommendation},  
  author={Park, Jin-Duk and Shin, Yong-Min and Shin, Won-Yong},  
  booktitle={Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval},  
  pages={2672--2676},  
  year={2024}  
} 
