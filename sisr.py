import numpy as np
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split

class SuperResolution:
    def __init__(self, sr_method, channels):
        self.sr_method = sr_method
        self.channels = channels

        data = np.load('./Training_1959-2015_Scaled.npz')
        if self.channels == 'RF':
            self.input_shape = (32,32,1)
            self.output_shape = (128,128,1)
            lr_inp = data['lr_inputs'][:,:,:,0:1]
            hr_out = data['hr_outputs']
        elif self.channels == 'RF+TOPO':
            self.input_shape = (32,32,2)
            self.output_shape = (128,128,1)
            lr_inp = data['lr_inputs'][:,:,:,0:2]
            hr_out = data['hr_outputs']
        elif self.channels == 'RF+WIND':
            self.input_shape = (32,32,2)
            self.output_shape = (128,128,1)
            lr_inp = np.stack((data['lr_inputs'][:,:,:,0], data['lr_inputs'][:,:,:,2]), axis = 3)
            hr_out = data['hr_outputs']
        elif self.channels == 'RF+TOPO+WIND':
            self.input_shape = (32,32,3)
            self.output_shape = (128,128,1)
            lr_inp = data['lr_inputs']
            hr_out = data['hr_outputs']
        else:
            raise ValueError('Invalid "channels" input. Choose either RF, RF+TOPO, RF+WIND, RF+TOPO+WIND.')
        
        del data

        
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(lr_inp, hr_out, train_size = 0.75, test_size=0.25, shuffle = True)
        del lr_inp, hr_out

    ###### ResBlocks and UpsamplingBlocks ######
    def res_block_srdrn(self, x0):
        x = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=1, padding='same')(x0)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.PReLU()(x)
        x = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=1, padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        return tf.keras.layers.Add()([x0, x])

    def upsampling_block_srdrn(self, x):
        x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same')(x)
        x = tf.keras.layers.UpSampling2D(size=(2,2), interpolation='nearest')(x)
        return tf.keras.layers.PReLU()(x)

    def res_block_edsr(self, x, num_filters):
        y = tf.keras.layers.Conv2D(num_filters, 3, padding='same')(x)
        y = tf.keras.layers.Activation('relu')(y)
        y = tf.keras.layers.Conv2D(num_filters, 3, padding='same')(y)
        y = tf.keras.layers.Add()([y, x])
        return y

    def Build_Autoencoder_ANN(self):
        """
        Autoencoder ANN Architecture
        """
        # Define the input and output layers
        inputs = tf.keras.layers.Input(shape=self.input_shape)
        x = tf.keras.layers.UpSampling2D(size = (4,4), interpolation = 'bilinear')(inputs)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(1024, activation='relu')(x)
        x = tf.keras.layers.Dense(512, activation='relu')(x)
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        x = tf.keras.layers.Dense(512, activation='relu')(x)
        x = tf.keras.layers.Dense(1024, activation='relu')(x)
        x = tf.keras.layers.Dense(128*128*1, activation='linear')(x)
        outputs = tf.keras.layers.Reshape((128,128,1))(x)

        return tf.keras.models.Model(inputs, outputs)
        
    def Build_SRCNN(self):
        """
        SRCNN Architecture
        """
        # Define the input layer
        inputs = tf.keras.layers.Input(shape=self.input_shape)

        # Add the feature extraction layers
        x = tf.keras.layers.Conv2D(filters=64, kernel_size=9, strides=1, padding='same', activation = 'relu')(inputs)

        # Add the non-linear mapping layers
        x = tf.keras.layers.Conv2D(filters=32, kernel_size=1, strides=1, padding='same', activation = 'relu')(x)

        # Add the upsampling feature maps layers
        x = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=4, strides=4, padding='same', activation = 'relu')(x)

        # Add the reconstruction layers
        outputs = tf.keras.layers.Conv2D(filters=1, kernel_size=5, strides=1, padding='same', activation = 'linear')(x)
        
        return tf.keras.Model(inputs, outputs)

    def Build_VDSR(self):
        """
        VDSR Architecture
        """

        # Define the input layer
        inputs = tf.keras.layers.Input(shape=self.input_shape)

        # Add the VDSR layers
        conv1 = tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu')(inputs)
        conv2 = tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu')(conv1)
        conv3 = tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu')(conv2)
        conv4 = tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu')(conv3)
        conv5 = tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu')(conv4)
        conv6 = tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu')(conv5)
        conv7 = tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu')(conv6)
        conv8 = tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu')(conv7)

        # Add the upsampling feature maps layers
        conv9 = tf.keras.layers.Conv2DTranspose(64, (4,4), strides=4, padding='same', activation = 'relu')(conv8)

        # Add the final output layer
        outputs = tf.keras.layers.Conv2D(1, (3,3), padding='same', activation='linear')(conv9)

        return tf.keras.Model(inputs=inputs, outputs=outputs)

    def Build_SRDRN(self):
        """
        SRDRN Architecture
        """
        inp = tf.keras.layers.Input(shape=self.input_shape)
        x0 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=1, padding='same')(inp)
        xi = x = tf.keras.layers.PReLU()(x0)
        
        for _ in range(4):
            x = self.res_block_srdrn(x)
        
        x = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=1, padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Add()([x, xi])
        
        for _ in range(2):
            x = self.upsampling_block_srdrn(x)
        
        x = tf.keras.layers.Conv2D(filters=1, kernel_size=3, strides=1, padding='same')(x)
        return tf.keras.Model(inputs=inp, outputs=x)


    def Build_EDSR(self):
        """
        EDSR Architecture
        """
        inputs = tf.keras.layers.Input(shape=self.input_shape)
        x = inputs

        # Pre-processing
        x = tf.keras.layers.Conv2D(64, 3, padding='same')(x)

        # Residual blocks
        for _ in range(16):
            x = self.res_block_edsr(x, 64)

        # Post-processing
        x = tf.keras.layers.Conv2D(3, 3, padding='same')(x)
        x = tf.keras.layers.Add()([x, inputs])
        x = tf.keras.layers.Activation('tanh')(x)

        # Upscaling
        x = tf.keras.layers.UpSampling2D(4)(x)

        x = tf.keras.layers.Conv2D(1,3,padding = 'same')(x)

        return tf.keras.models.Model(inputs, x)

    def reset_seeds(self):
        np.random.seed(1)
        np.random.seed(2)
        if tf.__version__[0] == '2':
            tf.random.set_seed(3)
        else:
            tf.set_random_seed(3)
        print("RANDOM SEEDS RESET")

    def Model_Training(self, expid, epochs, learning_rate, decay_rate, batch_size, model_save_path, csv_logger_path, restart, initial_epoch):
        run_code = expid+"_"+self.sr_method
        if ~os.path.exists(model_save_path+'/'+run_code):
            os.makedirs(model_save_path+'/'+run_code, exist_ok=True)
        if ~os.path.exists(csv_logger_path):
            os.makedirs(csv_logger_path, exist_ok=True)
        model_save_path = model_save_path+'/'+run_code
        csv_logger_path = csv_logger_path+'/'+run_code
        print("\t\t\t"+run_code+"\t\t\t")
        print(60*'-')
        # choose model based on sr_method input
        if restart == True:
            model = tf.keras.models.load_model(model_save_path+"/"+expid+"_"+self.sr_method+"_best_model.h5", compile = False)
            print('Model reloaded, epoch: '+str(initial_epoch-1))
        else:
            if self.sr_method == 'SRCNN':
                model = self.Build_SRCNN()
            elif self.sr_method == 'SRDRN':
                model = self.Build_SRDRN()
            elif self.sr_method == 'VDSR':
                model = self.Build_VDSR()
            elif self.sr_method == 'EDSR':
                model = self.Build_EDSR()
            elif self.sr_method == 'AUTOENC':
                model = self.Build_Autoencoder_ANN()
            else:
                raise ValueError('Invalid sr_method input. Choose either SRCNN or SRDRN')
        # Defining Callbacks
        def exp_decay(epoch):
            initial_lrate = learning_rate
            k = decay_rate
            lrate = initial_lrate * np.exp(-k*epoch)
            return lrate

        lrate = tf.keras.callbacks.LearningRateScheduler(exp_decay)     
        modelsave_callback = tf.keras.callbacks.ModelCheckpoint(filepath = model_save_path+"/"+expid+"_"+self.sr_method+"_best_model.h5", initial_epoch = initial_epoch, monitor='val_root_mean_squared_error', verbose=1, save_best_only=True, save_weights_only=False, mode='auto')
        training_csv_logger = tf.keras.callbacks.CSVLogger(csv_logger_path+"/"+expid+"_"+self.sr_method+"-training-log.csv", append = True)
        estop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)
        # compile the model
        opti = tf.optimizers.Adam(learning_rate = learning_rate)
        model.compile(optimizer=opti, loss='mse', metrics=[tf.keras.metrics.RootMeanSquaredError()])
        # train the model
        history = model.fit(self.x_train, self.y_train, epochs=epochs, validation_data=(self.x_test,self.y_test), batch_size=batch_size, verbose=2, callbacks=[modelsave_callback,training_csv_logger, estop, lrate])
        tf.compat.v1.reset_default_graph()
        tf.keras.backend.clear_session()
        del model, modelsave_callback, training_csv_logger, history

