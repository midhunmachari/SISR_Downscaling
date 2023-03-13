from sisr import SuperResolution
import tensorflow as tf
exp_id = 'EXP001'
sr_method = 'EDSR'
epochs = 2500
learning_rate = 10**-5
decay_rate = 0.0025
batch_size = 128
channels = 'RF+TOPO+WIND'
model_save_path = './MODEL_RUNS'
csv_logger_path = './MODEL_RUNS'

sr = SuperResolution(sr_method = sr_method, channels = channels)
sr.reset_seeds()
sr.Model_Training(expid = exp_id, epochs = epochs, learning_rate = learning_rate, 
                    decay_rate = decay_rate, batch_size = batch_size, model_save_path = model_save_path, 
                    csv_logger_path = csv_logger_path, restart = False, initial_epoch = 0)

tf.compat.v1.reset_default_graph()
tf.keras.backend.clear_session()
