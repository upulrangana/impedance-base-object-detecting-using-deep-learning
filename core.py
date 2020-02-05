# multi-class classification with Keras
import pandas
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder


# define baseline model
def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(8, input_dim=105, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def get_prediction():
    # load dataset
    dataframe = pandas.read_csv("csv/mini_sample.csv", header=None)
    dataset = dataframe.values
    X = dataset[:, 0:105].astype(float)
    Y = dataset[:, 105]
    # encode class values as integers
    encoder = LabelEncoder()
    encoder.fit(Y)
    encoded_Y = encoder.transform(Y)
    # convert integers to dummy variables (i.e. one hot encoded)
    dummy_y = np_utils.to_categorical(encoded_Y)

    estimator = KerasClassifier(build_fn=baseline_model, epochs=200, batch_size=5, verbose=0)
    kfold = KFold(n_splits=10, shuffle=True)
    results = cross_val_score(estimator, X, dummy_y, cv=kfold)
    print("Baseline: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))

    dataframe = pandas.read_csv("csv/predict.csv", header=None)
    dataset = dataframe.values
    Xnew = dataset[:, 0:105].astype(float)
    ynew = baseline_model().predict_classes(Xnew)
    # show the inputs and predicted outputs
    # print("X=%s, Predicted=%s" % (Xnew[0], encoder.inverse_transform(ynew)[0]))
    return encoder.inverse_transform(ynew)[0]
