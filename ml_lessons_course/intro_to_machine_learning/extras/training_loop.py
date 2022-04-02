# -----------------------------------------------------------
# Example of custom training loop
#
#
# dmike16, Rome, Italy
# Released under MIT license
# email cipmiky@gmail.com
# -----------------------------------------------------------

import bag.line_gaussian_noise as lgn
import bag.lowlevels.simple_linear_regression as slr
import bag.lowlevels.loss_functions as loss
import bag.lowlevels.optmizer_functions as opt
import matplotlib.pyplot as plt
import tensorflow as tf

# Create the input data
TRUE_W = 3.0
TRUE_B = 2.0
ds_line = lgn.LineGussianNoise(TRUE_W, TRUE_B, num_example=201)

ds_line.plot()
model = slr.SimpleLinearRegression()
# the model watch the variables
print("Variables {}".format(model.variables))
# verify tha the model work
assert model(3.0).numpy() == 15.0
# Current loss without training
print("Current loss {:1.6f}".format(loss.mse(ds_line.y, model(ds_line.x))))
ds_line.plot_with_predictions(model(ds_line.x))
epochs = range(10)
history = model.train(epochs=len(epochs), target=ds_line.y, x=ds_line.x, learning_rate=0.1, optimizer=opt.sdg,
                      loss=loss.mse)
plt.plot(epochs, history['w'], label='Weights', color='blue')
plt.plot(epochs, [TRUE_W] * len(epochs), '--',
         label="True weight", color='blue')

plt.plot(epochs, history['b'], label='bias', color='red')
plt.plot(epochs, [TRUE_B] * len(epochs), "--",
         label="True bias", color='red')

plt.legend()
plt.show()
ds_line.plot_with_predictions(model(ds_line.x))

kmodel = slr.SimpleLinearRegressionKeras()
kmodel.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.1),
    loss=tf.keras.losses.mean_squared_error,
)
kmodel.fit(ds_line.x, ds_line.y, epochs=len(epochs), batch_size=1000)
