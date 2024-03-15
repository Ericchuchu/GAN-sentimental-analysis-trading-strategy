import tensorflow as tf
from tensorflow.keras.utils import plot_model
from model import make_discriminator_model,make_generator_model
from data_preprocess import X_train,X_test,y_train,y_test,input_dim,output_dim,yc_train,yc_test,index_test
from train import train,eval_op
from visualization import plot_test_data

learning_rate = 5e-4
epochs = 500

g_optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate)
d_optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate)

generator = make_generator_model(X_train.shape[1], output_dim, X_train.shape[2])
discriminator = make_discriminator_model(X_train.shape[1])

# plot model
'''plot_model(generator, to_file='generator_keras_model.png', show_shapes=True)
tf.keras.utils.plot_model(discriminator, to_file='discriminator_keras_model.png', show_shapes=True)'''

predicted_price, real_price, RMSPE = train(X_train, y_train, yc_train, epochs, generator, discriminator, g_optimizer, d_optimizer)
test_generator = tf.keras.models.load_model(f'AMD_generator_V_{epochs-1}.keras')
predicted_test_data = eval_op(test_generator, X_test)
plot_test_data(y_test, predicted_test_data,index_test)