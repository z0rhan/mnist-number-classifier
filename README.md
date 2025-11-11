# About

This is my attempt at writing a deep neural network from scratch for the MNIST digit classification problem.

You can find the dataset that I'm using here:  
[MNIST Original on Kaggle](https://www.kaggle.com/datasets/avnishnish/mnist-original)

# Running

You can install the required packages required by:
```bash
pip install -r requirements.txt
```

Then you can train and test model with:
```bash
python3 main.py
```

# Iteration 1

(Results in this section correspond to commit [`dab5dae`](https://github.com/z0rhan/mnist-number-classifier/tree/dab5dae8e4781b90d93cc916617cd14b89fb4b9f).)

This is the first iteration of my model.

- Architecture: 2 hidden layers with 10 nodes each
- Output: 1 output neuron
- Activation function: `tanh` for the hidden layers and output

This setup is inspired by one of the exercises from the deep learning course I’m taking at my university, but the results were quite poor.  
When testing on the **training data itself**, I only got an accuracy of **~9%**.

This is not surprising: I’m trying to do **10-class classification with only a single output node**. The model effectively ends up biased toward the class that appears last during training, which matches the accuracy we observed.

So for the next iteration, I'm going to increase the output neurons to 10. Let's see how far can we get.
