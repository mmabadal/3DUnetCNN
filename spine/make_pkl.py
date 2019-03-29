import pickle

val = open('validation_ids.pkl', 'rb')
info_val = pickle.load(val)

training = open('training_ids.pkl', 'rb')
info_train = pickle.load(training)

# save train data in a file
train = open('training_ids_2.pkl', 'wb')
pickle.dump(info_train, train, protocol=2)
train.close()

# save val data in a file
val = open('validation_ids_2.pkl', 'wb')
pickle.dump(info_val, val, protocol=2)
val.close()
