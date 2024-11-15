# Large Language Model (LLM) from scratch using Python

### Acknowledgements
I would like to extend my sincere thanks to **[freeCodeCamp](https://www.freecodecamp.org/)** and the creator of the **[tutorial](https://www.youtube.com/watch?si=bKUnRhoGHbYjT5Ay&v=UU1WVnMk4E8&feature=youtu.be)** for their invaluable content and guidance in helping me build this project. This project wouldn't have been possible without their educational resources.

---

<br>
<br>

## Introduction : What is an LLM ? 🤔

Imagine you’re texting a friend, and they ask :<br>

**"Hey, do you know how to bake banana bread?"**<br>

You pause... baking isn't exactly your strong suit. But instead of admitting defeat, you turn to your trusty LLM (Large Language Model), which instantly gives you the perfect banana bread recipe — with tips on how to make it extra fluffy. 🍞🍌
<br>
<br>

Crisis averted! Thanks to the LLM, you’re now the banana bread expert your friend thinks you are.

---

<br>

Well, that's a small funny introduction of one of the use cases of LLMs. Now, let's dive into the technical world of LLMs !
<br>

Large Language Models are the backbone of many advanced AI systems today. They are trained on vast amounts of text data, enabling them to understand, generate, and even engage in meaningful conversations with users. In this project, we'll explore how LLMs work from the ground up, covering everything from tokenization and embeddings to training techniques and model fine-tuning.

<br>

According to **AWS**🔻
> **Large Language Models (LLMs)** are advanced deep learning models pre-trained on massive datasets, typically utilizing the transformer architecture. Transformers consist of an encoder and decoder with self-attention mechanisms, enabling them to understand relationships between words and extract meanings from text sequences. Unlike earlier RNNs that process inputs sequentially, transformers handle entire sequences in parallel, making them more efficient and faster to train using GPUs. Their architecture supports massive models, often with hundreds of billions of parameters, allowing them to ingest vast amounts of data from sources like Common Crawl and Wikipedia, learning grammar, language, and knowledge through self-learning.

<br>

Now that you have an introductory knowledge about **LLMs**, it's time to dive into the components of this repo 🔻<br>

> **NOTE** : Because the notebook size are pretty big - you may not be able to render the notebook / code blocks. Thus, it is better to ***fork*** the repository / download the individual files to go through them.

<br>
<br>

### [PyTorch Basic Functions](https://github.com/sricks404/LLM-from-scratch-using-Python/blob/main/pytorch_basic_funcs.ipynb) 👇 <br>

This notebook contains all the necessary basic functions (from PyTorch) needed to build the LLM.

<br>
<br>

### [PyTorch CUDA GPU](https://github.com/sricks404/LLM-from-scratch-using-Python/blob/main/pytorch_CUDA_GPU.ipynb) 👇 <br>

Since the final LLM model will be trained on the **[OpenWebText](https://huggingface.co/datasets/Skylion007/openwebtext)** dataset, which was also used to train the **GPT-2** model, training on a CPU would take an extremely long time ☠️. Instead of using a CPU, we trained the model on a GPU (Graphical Processing Unit). The GPU used here is the **NVIDIA GeForce GTX 1050**.<br><br>

Incorporating GPU acceleration (NVIDIA CUDA) into model training can be challenging. This notebook gives you an idea how fast GPUs are as compared to the CPUs.

<br>

> **NOTE** : Each notebook in this repository is thoroughly commented, ensuring that you understand the "_Why's and How's_". The comments will provide a clear explanation as you go through the code.

<br>
<br>

### [Bi-Gram](https://github.com/sricks404/LLM-from-scratch-using-Python/blob/main/bigram.ipynb) 👇 <br>

A **`Bi-Gram Model`** is a straightforward yet powerful tool in Natural Language Processing (NLP) and language modeling. It works by looking at pairs of consecutive words in a text to predict what comes next. For example, if we consider the sentence "I love cats," the bi-grams would be "I love" and "love cats." By examining how often these word pairs appear together, the model learns the likelihood of a word following another. This helps in generating text that feels natural. However, while bi-gram models capture basic word relationships, they only consider immediate word pairs, so they might miss out on more complex or longer-term patterns in language. Despite this, they’re a key building block for understanding how words fit together and are often used as a foundation for more advanced models.
<br>
<br>
*Keywords : Batch Size, Block Size, Encoder-Decoder, Tensors, Optimizers*
<br>
<br>
This **Bi-Gram Model** is trained on the dataset (or rather, a book) sourced from **Project Gutenberg: [The Adventures of Sherlock Holmes](https://www.gutenberg.org/ebooks/1661)**

> **NOTE** : The focus of this project was on building the model, not on data preprocessing, so that step has been omitted.

<br>
<br>

### [GPT - Language Model (Version #1)](https://github.com/sricks404/LLM-from-scratch-using-Python/blob/main/gpt-v1%20-%201.ipynb) 👇<br>

Now that you have explored the implementation of the **Bi-Gram** model, we can take the first step in building our initial LLM model: _**GPT-v1 (Version 1)**_.

> **NOTE**  : This GPT model is also trained on the book: [The Adventures of Sherlock Holmes](https://www.gutenberg.org/ebooks/1661). In a subsequent notebook, you will see the model trained on the **OpenWebText** dataset.

In this notebook, even though the model is still based on the **Bi-Gram Model**, we can see that the hyperparameters used are now different from what we saw in the **Bi-Gram Model Implementation**. This is because we are trying to build a **Transformer**, which is the primary building block for GPT model(s). Let us list them all here🔻

 - **block_size**: The length of input sequences the model processes at once (e.g., 32 to 128 tokens).
    
-   **batch_size**: Number of samples processed before updating the model weights (e.g., 64 to 512).
    
-   **max_iters**: Total number of training iterations (e.g., 1000 to 5000).
    
-   **learning_rate**: The step size for updating model weights, with potential decay over time (e.g., 1e-5 to 1e-3).
    
-   **eval_iters**: Frequency of evaluating the model during training (e.g., every 50 to 500 iterations).
    
-   **n_embd**: Dimensionality of the embedding vectors (e.g., 128 to 512).
    
-   **n_layer**: Number of layers in the model architecture (e.g., 6 to 12).
    
-   **n_head**: Number of attention heads in each layer (e.g., 4 to 12).
    
-   **dropout**: Proportion of neurons randomly set to zero to prevent overfitting (e.g., 0.1 to 0.3).

<br>

> **NOTE** : Even though the dataset is relatively small (just a book) compared to what we will use in the next version of our GPT model, and the model is being trained on a GPU, the training and validation process may still take a significant amount of time. This is highly dependent on the values chosen for each hyperparameter. Therefore, be patient 🙂 (and not 🤪) — results will take time to show up ⌛.

<br>

The main goal of our GPT model (for now) is to achieve the lowest Validation Loss and generate content that appears relevant and meaningful. Don’t get disheartened if you encounter some "alien-like" language being generated 👽.

<br>
<br>

### [GPT - Language Model (Version #2)](https://github.com/sricks404/LLM-from-scratch-using-Python/blob/main/gpt-v1%20-%202.ipynb) 👇<br>

Alright folks, it's time to dive into some serious stuff. The dataset we'll be using is **[OpenWebtext](https://huggingface.co/datasets/Skylion007/openwebtext)**. This is the same dataset on which **[OpenAI's GPT-2](https://openai.com/index/better-language-models/)** was trained. 

<br>

> **NOTE** : Due to the enormous size of the dataset and limited computing capacity, we've selected appropriate and "healthy" hyperparameter values to avoid overloading your system. Using inappropriate values could result in 💣💥☠️💀👻. Please use the appropriate settings to ensure a smooth and efficient process !

<br>

The **OpenWebText** dataset is quite different from the usual datasets we might download from websites like Kaggle or the UC-Irvine data repository. Instead, it's a ZIP file containing multiple sub-folders, each with numerous other ZIP files inside. Although some manual effort was involved in organizing the data, I still needed to handle the ***[Data Extraction](https://github.com/sricks404/LLM-from-scratch-using-Python/blob/main/DataExtract.py)*** component to ensure the data is fed into our GPT model in the correct format.

> **NOTE** : Jupyter Notebooks tend to take more time in model training compared to when the code is reproduced in the format of a Python script file (**[.py]** file). Thus, the model training code is also available in the **[training.py](https://github.com/sricks404/LLM-from-scratch-using-Python/blob/main/training.py)** file. Another reason for creating this Python file is that you cannot call a **Command Line Interface** inside a Jupyter Notebook.<br><br>
**Why?** 👇  
In Jupyter Notebooks, the `argparse` library is not typically suited for direct use because it is designed to parse command-line arguments, and Jupyter itself starts with its own command-line arguments that may conflict with those defined by `argparse`. When you try to use `argparse` in a notebook, it attempts to parse Jupyter’s internal arguments, which can result in errors.
>

<br>

Later in this notebook, we've implemented a method to save the trained model. This allows us to load the model in a separate Python script. Through the command-line interface ( Code : ***[cmdLineParsing](https://github.com/sricks404/LLM-from-scratch-using-Python/blob/main/cmdLineParsing.py)*** ), we can then dynamically provide the ***batch_size*** hyperparameter value when generating new text.

<br>
<br>

### [Creating a Dummy ChatBot on CLI](https://github.com/sricks404/LLM-from-scratch-using-Python/blob/main/chatbot.py) 👇<br>


A GPT model can be utilized in many forms. Today, numerous websites have integrated chatbots to make life easier for their users, helping them get answers to questions and complete tasks with less time and effort.

<br>

In this instance, we will demonstrate a chatbot functioning through the **Command Line Interface (CLI)**, where it will prompt the user for input, and the GPT model will generate a response.

<br>

> **NOTE** : The GPT model is created and trained with a general purpose, so its responses might occasionally seem less relevant or specific. However, incorporating **[Retrieval-Augmented Generation (RAG)](https://cloud.google.com/use-cases/retrieval-augmented-generation)** can significantly enhance the chatbot's functionality, allowing it to serve as a more effective virtual assistant. This enhancement provides better access to relevant information and improves the chatbot's ability to meet user needs, especially when it is trained on domain-relevant data.

<br>
<br>

### [Gradio Application](https://github.com/sricks404/LLM-from-scratch-using-Python/blob/main/GradioApp.py) 👇<br>
Last, but not least – since we’ve created a **ChatBot**, it’s time to deploy it rather than just passing CLI arguments. In this guide, we will use **[Gradio](https://www.gradio.app/)** to deploy our ChatBot. Gradio will allow us to create a user-friendly interface where🔻

1.  **API Application** : The application will receive a **prompt from the user**.
2.  **Output** : It will generate and display **text** based on the input.

<br>
<br>


## **Running Locally** 👇

### **1. Clone the Repository**

`git clone https://github.com/SoubhikSinha/LLM-from-scratch-using-Python.git` 

### **2. Create a Virtual Environment**

You can either create the virtual environment in the same directory or use a central Anaconda environment directory:

-   **Option 1**:
    
    `conda create --name LLM python=3.11` 
    
-   **Option 2**:
        
    `conda create --prefix ./LLM python=3.11` 
    

### **3. Activate the Conda Environment**

`conda activate ./LLM` 

### **4. Install Required Libraries**

`pip install -r requirements.txt` 

### **5. Register a Jupyter Notebook Kernel**

`python -m ipykernel install --user --name=LLM --display-name "LLM-cuda-gpt"` 

This will allow you to use the `LLM` environment in Jupyter Notebook. Choose the `"LLM-cuda-gpt"` kernel when running notebooks in your editor (e.g., VS Code).

### **6. Set Up GPU Support for PyTorch**

If your system has an NVIDIA GPU, follow this tutorial to enable GPU support for PyTorch:  
[How to setup NVIDIA GPU for PyTorch on Windows 10/11](https://www.youtube.com/watch?v=r7Am-ZGMef8&t=381s).

> **Note:** Ensure GPU integration is configured in the same Conda environment (`LLM`).

----------

## **Suggested Workflow**

Follow the notebooks in the order below for a structured understanding of the concepts:

1.  **`pytorch_basic_funcs.ipynb`**: Learn the basic PyTorch functions.
2.  **`pytorch_CUDA_GPU.ipynb`**: Understand how to leverage GPU for PyTorch operations.
3.  **`bigram.ipynb`**: Explore the working of a Bi-Gram model.
4.  **`gpt-v1_1.ipynb`**: Dive into the architecture of GPT using the "Attention is All You Need" paper. [[Paper Link](https://arxiv.org/abs/1706.03762)]

----------

## **Working with OpenWebText Dataset**

To train the GPT model on a large dataset like **[OpenWebText](https://huggingface.co/datasets/Skylion007/openwebtext)**:

1.  **Prepare the Dataset**:
    
    -   Download and organize the dataset ZIP files into the `OpenWebText` folder.
    -   Ensure all files named `urlsf_subset*` are sequentially arranged in the folder.
2.  **Extract the Data**:
    
    -   Run the data extraction program:
              
        `python DataExtract.py` 
        
    -   This generates three files: `train.txt`, `val.txt`, and `vocab.txt`.
    
    > **Note:** Ensure the path to the `OpenWebText` folder is correctly set in the script.
    
3.  **Train GPT**:
    
    -   Open and run the notebook `gpt_v1-2.ipynb`.

----------

## **Interactive Chatbot**

### **Command-Line Interface (CLI)**

Run the chatbot with a specified batch size:

`python chatbot.py -batch_size 32` 

> You can experiment with batch sizes like `16`, `32`, or `64`.

### **Gradio Interface**

-   Run the Gradio app:
        
    `python GradioApp.py` 
    
-   Open the IP address provided (e.g., `http://127.0.0.1:7860`) to access the GPT model in a minimalistic web interface.
    
-   Optionally, deploy your model to [HuggingFace](https://huggingface.co/) for broader access.
    

----------

## **Key Notebooks**

-   **`pytorch_basic_funcs.ipynb`**: Introduction to PyTorch.
-   **`pytorch_CUDA_GPU.ipynb`**: GPU integration with PyTorch.
-   **`bigram.ipynb`**: Bi-Gram model fundamentals.
-   **`gpt-v1_1.ipynb`**: GPT architecture basics.
-   **`gpt_v1-2.ipynb`**: Training GPT with OpenWebText.

<br>

### And that said, we are done with our LLM (GPT) model implementation ! <br>
### 🙌🎉🥳🎊
