import pickle

model = None
with open('modeling.pkl','rb') as pickle_file:
   model = pickle.load(pickle_file)

X_test = [[6,148,72,35,0,33.6,0.63,50]]
y_pred = model.predict(X_test)

print(y_pred)