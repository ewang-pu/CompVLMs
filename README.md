## CompGPT: Probing the Compositional Understanding of Visual Language Models

# Overview
This repository contains the code and datasets used in the independent research project titled "CompGPT: Probing the Compositional Understanding of Visual Language Models" by Evan Wang under the advisement of Felix Heide. The research focuses on the limitations of Vision Language Models (VLMs) in understanding compositional relations, i.e., how elements in an image relate to each other. This project extends evaluations on new models, investigating current methods to evaluate and improve compositional understanding in VLMs, and refining these methods to contribute new data and directions on the question of compositional awareness in VLMs.

<!-- # Installation
To get started with this project, clone the repository and install the required dependencies:

bash
Copy code
git clone https://github.com/ewang-pu/CompVLMs.git -->

# Usage
This repository is structured to facilitate the evaluation and improvement of compositional understanding in VLMs. For a detailed guide on how to run the evaluations and utilize the datasets, refer to the Documentation section.

Acknowledgements
Special thanks to Dr. Felix Heide for his invaluable guidance throughout the research, as well as to all contributors and participants in the study for their efforts to advance the understanding of compositional understanding in visual language models.

# Research Paper
The full reseach paper can be viewed here: [Link to my paper](CompGPT_Report.pdf)
For a tldr:
* The Vision-and-Language Transformer, even with word patch alignment training, does not perform much better than other VLMs on compositional reasoning tasks.
* Existing datasets for evaluating compositional reasoning are biased: "blind" models can outperform VLMs via perplexity calculations
* LLMs with natural language understanding provide a method of reducing these biases
