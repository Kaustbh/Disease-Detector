# Disease-Detector


## About
Predicting whether you have the disease, based on the input features provided.
The Diseases Predicted are :

1. Breast Cancer ( Based on the input ,it predicts whether the paitient has "Malignant" tumor or "Benign" tumor)

2. Parkinson Disease

3. Diabetes

## Requirements :

 Install all the dependencies:

```
pip install -r requirements.txt
```

## Discription :

First I have trained all the Diseases Separetly , and saved their models. Then I imported this 3 models in "MultipleDisease.py" 
and made the GUI of the Predictive System using Streamlit .

What can Streamlit do?

Streamlit is an open source app framework in Python language. It helps us create web apps for data science and machine learning in a short time. It is compatible with major Python libraries such as scikit-learn, Keras, PyTorch, SymPy(latex), NumPy, pandas, Matplotlib etc.

You can run this python file wiht the command as "streamlit run MultipleDisease.py"  ,  then you will be directed to the localhost webpage where you can test the Predictive System.

![disease](https://user-images.githubusercontent.com/97254178/213727019-a016ce20-9e41-4fea-9029-605134db9034.png)


