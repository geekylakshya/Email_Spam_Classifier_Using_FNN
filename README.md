# Email Spam Classifier Project

This project aims to build a machine learning model to classify emails as spam or ham (not spam) using natural language processing techniques and TensorFlow.

## Dataset

The dataset used in this project is a collection of emails labeled as spam or ham. It contains two columns: `Category` (spam/ham) and `Message` (email text). The dataset is stored in a CSV file named `mail_data.csv`.

**Note:** The dataset may have been updated since the creation of this project. For the most recent version of the dataset, please refer to [this link](https://example.com/updated_mail_data.csv).

## Requirements 

To run this project, you'll need the following dependencies:

- Python 3.12
- TensorFlow
- TensorFlow Hub
- NumPy
- Pandas
- Matplotlib

You can install the required packages via pip:

```bash
pip install tensorflow tensorflow-hub numpy pandas matplotlib
```

## Model Architecture 

The model architecture used in this project is a feedforward neural network:

- **Input Layer:** Utilizes a TF-Hub layer to convert text into numerical embeddings.
- **Hidden Layers:** Two dense hidden layers with ReLU activation functions.
- **Output Layer:** Single neuron output layer with sigmoid activation function for binary classification.

The model is trained using binary cross-entropy loss and Adam optimizer.

## User Input

To add a fun twist and test the model's performance further, users can input their own email text to check whether it is classified as spam or ham.

### Instructions:

1. Swap out the `input_mail` variable with your own email text.
2. Run the code to predict whether the input email is spam or ham.

### Example Usage:

```python
input_mail = ["Your very own email text goes here."]
```
<small>And hey, just a heads up! While our model has been trained on a dataset of emails, it might miss out on catching some of the latest spam trends circulating these days. So, don't be surprised if it doesn't catch all the newest tricks up those sneaky spammers' sleeves! 😄📧🔍</small>

## Usage 

1. Clone the repository:

```bash
git clone https://github.com/geekylakshya/Email_Spam_Classifier.git
cd Email_Spam_Classifier
```
2. Run the Jupyter Notebook Email_Spam_Classifier.ipynb to train the model and evaluate its performance.

3. After training the model, you can use it to classify new emails. Modify the input_mail variable in the notebook with the email text you want to classify, then run the cell containing the prediction code.

## Evaluation 

The model's performance is evaluated on a separate test set, as well as on the training and validation sets during training. The evaluation metrics include accuracy and loss.

- Accuracy : 95%+

## Results 

The project provides insights into the effectiveness of using TensorFlow and natural language processing techniques for email spam classification. Results are visualized using Matplotlib.


## Author

Made By Me (**Lakshay**)

## License 

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
