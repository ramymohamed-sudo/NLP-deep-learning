# here, closer look to the weights and the overfitting 


# LSTM: return state and return sequence 



# in keras
model.train_on_batch(x, y)
weights = model.get_weights()[0]




model.fit(x=x_train, y=y_train, batch_size=128, epochs=20,
          validation_data=(x_test, y_test),
          verbose=1, callbacks=[early_stopping, save_best])

glove_embeds = model.layers[0].get_weights()[0]
