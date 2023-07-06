__[IT4772E-NLP]__

Capstone project for Natural Language Processing course, SoICT, HUST, Spring 2023. 

Implementation of CAt approach for Unsupervised Aspect Category Detection. The original method is proposed in the paper: "Embarrassingly Simple Unsupervised Aspect Extraction" by Tulkens and Cranenburgh, 2020. 


## 1. Settings

  - 1.1 Download the dataset: 
    
    The dataset can be downloaded by [SemEval-2014](https://drive.google.com/drive/folders/14Gl9ZKI4hptVEJc6qYrR8N0AomfhF71k?usp=sharing) and [CitySearch](https://drive.google.com/drive/folders/122W9h6bkZ1xPdabgp456vP3rnIyOuFpW?usp=sharing). 
    
    Put the SemEval-2014 dataset in `data/semeval2014/` folder, and CitySearch dataset in `data/citysearch/` folder.

  - 1.2 Preprocessing. Run: 

    `python embeddings/preprocessing.py`
## 2. Experiments
  - After running `preprocessing.py`, you are able to evaluate your text by go to part _2.2_ or re-experiment of 2 datasets CitySearch or SemEval-2014 restaurant by go to part _2.1_. 

  - 2.1 Experiment yourself on CitySearch dataset or SemEal-2014 restaurant dataset .
    
    `python main.py`


  - 2.2 Test your own experiment in restaurant domain (change your text in the file before running).
  
    `python inference.py`

