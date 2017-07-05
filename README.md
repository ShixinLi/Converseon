Instruction:

Run the program under Python 2.7

To run the program, simply enter python Main.py, then you will be asked to enter the path of your raw data, for example iris.data.txt as given. After that, my program will evaluate 5 different multi-class classification models, including logistic regression, and choose the best one to train based on the leave one out cross-validation method. After training the model, you will be then asked to enter the path of the test file in json format. Enter the path of the test file in json format, and you will see the output.


Bonus Questions:

1. Yes, logistic regression can do the multi-class classification. I used sklearn.linear_model.LogisticRegression, and changed the parameter multi_class from default:'ovr' to 'multinomial'. Also, I changed the parameter solver from default:'liblinear' to 'newton-cg', since 'newton-cg' is for multiclass.

  If you want to use logistic regression, you should open Main.py file and change the trainer.train(input_data) to               trainer.train(input_data, algo=lr (or anything else). Logistic regression will be trained when the argument 'algo' is not     None.


2. In order to evaluate the classifiers, I used leave one out cross-validation strategy. By applying this method, the number of fold equals to the size of the data (fold = len(y) or len(X)). We then can use the mean of the scores to compare different models and find out the best one. Since our data is very small, it's better to use leave one out strategy to try to use the most data to train the model and get the most persuasive accuracy. Also, since our data is very small, I think keep default is enough.
