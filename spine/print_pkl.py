import pickle

training = open('training_ids.pkl', 'rb')
info_train = pickle.load(training)
print("train:" + str(info_train))
val = open('validation_ids.pkl', 'rb')
info_val = pickle.load(val)
print("val:" + str(info_val))

