# Product-Recommendation-System-2023

This project is the coursework for 2022-2023 study year. The idea behind the project was to create ontology based on the Klink algorithm.

final_amazon_klink has the demonstratation on workings of all the functions and modules.

Amazon_preprocess has two classes: DataSetPreprocessor and TextPreprocessor - necessary for preprocessing of the dataset used in the project.

creation has two functions: create_ontology and visualise_ontology for creating and visualising the created ontology respectively.

preprocessed_df.csv is the dataset that was created as the result of DataSetPreprocessor and TextPreprocessor working on the first 250000 products in the Amazon metadata dataset.

ontology.csv is the dataset that was created as a results of function create_ontology with min_df = 0.015 from some of the data in the preprocessed_df.
