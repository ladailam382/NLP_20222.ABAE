# NLP_20222.AspectCategoryDetection

The repository consists of different approach for Aspect Category Detection problem. For each methods, please go inside your desired folders and explore it yourself. \
We also provide a Webapp for this project, which you can choose and test each method with your inputs. The following steps describe how you can launch our deployment yourself in your local workspaces.

## 1. Requirements


First, you need to install prerequirements, using the following command.

```
pip install -r requirements.txt
```

## 2. Download Dataset and Embeddings

- 1.1 Download the dataset: 
    
    The dataset can be downloaded by [SemEval-2014](https://drive.google.com/drive/folders/14Gl9ZKI4hptVEJc6qYrR8N0AomfhF71k?usp=sharing) and [CitySearch](https://drive.google.com/drive/folders/122W9h6bkZ1xPdabgp456vP3rnIyOuFpW?usp=sharing). 

- 1.2 Preprocessing. 
    
    For some methods, the embedding weights are heavy, so you need to process the data in some directories before you can launch the web application. For more detail, we highly recommend you to carefully read the instructions of each method in corresponding folder.

## 3. Deployment

To run our application, you just simply run following commands and enjoy:

```
gradio app.py
```


