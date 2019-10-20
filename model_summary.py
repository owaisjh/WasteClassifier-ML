from keras.models import load_model
model = load_model("trash_model.hdf5")
print(model.summary())
