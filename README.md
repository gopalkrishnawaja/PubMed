## Project Overview
This project is focused on optimizing the performance of a question-answering system. The system is trained on a dataset of medical abstracts and evaluated on the PubMedQA dataset. The project aims to find the best combination of hyperparameters, such as chunk size, top-k, and threshold, to maximize the accuracy of the system.

### About the files
`data`: This directory contains the dataset used for training and evaluation, including the test_ground_truth.json file.

`preprocess` : This directory has code for preprocessing and splitting the dataset.

`.env`: This file stores the OpenAI API key, which is loaded using the dotenv library.

`openAi_llm.py`: The Python script that loads the dataset, defines the objective function for hyperparameter optimization, and runs the ParamTuner and RayTuneParamTuner for OpenAI's `gpt-3.5-turbo` llm.

`gimini.py`: The Python script that loads the dataset, defines the objective function for hyperparameter optimization, and runs the ParamTuner and RayTuneParamTuner for Google's `gemini-pro` llm.

`test_mistral` : Started to implement experimentation of mistral but fell short of time.

`Results.docx` : Tabulates the results of hyperparameter tuning for openai and gemini.

`PubMedQA` : Contains the explanation of the project implementation