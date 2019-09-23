def run(process,embedding, origin_len, addFilter):
    import keras
    from keras.callbacks import EarlyStopping,Callback
    from keras.utils import plot_model
    from keras.utils.np_utils import to_categorical
    from keras.callbacks import LearningRateScheduler          
    from keras import optimizers
    from keras import backend as K
    from keras.preprocessing.image import ImageDataGenerator
    from keras.datasets import cifar10


    from model import model_fn
    from DNNcoder import DNNcoder as coder
    
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    mpl.use('Agg')
    

    import tensorflow as tf
    import random
    import heapq
    from contextlib import redirect_stdout
    import numpy as np
    import pickle
    import gc
    from sklearn import preprocessing
    from tensorflow.python.keras.applications.resnet50 import preprocess_input
    import six
    import sys
    from utils import random_crop, crop_generator, center_crop, crop_generator_center, m_images_pad, principal_components, zca_whitening, MyCallback, SGDRScheduler
    

    if process == 'searching':
        search = True   
    elif process == 'final':
        search = False
     
    
    #simulation data
    simulation_sequence = []
    num_data = 10

    for i in range(num_data):
      foo = [i for i in range(1, 30)]*4
      simulation_sequence.append(random.sample(foo, origin_len))

    simulation_sequence = np.reshape(simulation_sequence,[num_data, origin_len])
    s = random.sample(range(simulation_sequence.shape[0]),1)
    fake = simulation_sequence[s]

    del simulation_sequence, foo, s,             
    gc.collect()



    batch_size = 128
    episodesCount = 0
    
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = (x_train.astype('float32')-np.mean(x_train)) / np.std(x_train)
    x_test = (x_test.astype('float32')-np.mean(x_test)) / np.std(x_test)
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)


    #data pad and shuffer
    shuffer_ind = random.sample(range(x_train.shape[0]), x_train.shape[0])
    x_train = x_train[shuffer_ind]
    y_train = y_train[shuffer_ind]

    x_train = m_images_pad(x_train,4)
    x_test = m_images_pad(x_test,4)

    del shuffer_ind      
    gc.collect()    

    if search:      
        epochs = 10
        episodes  = 120
        train_rate = 0.9
        tr_datagen = ImageDataGenerator(
                        horizontal_flip=True,
                        validation_split=1-train_rate,
                        )
        tr_datagen.fit(x_train)  
        train_generator = tr_datagen.flow(x_train, y_train, batch_size=batch_size, subset='training')
        val_generator = tr_datagen.flow(x_train, y_train, batch_size=batch_size, subset='validation')
        train_crops = crop_generator(train_generator, 32)
        val_crops = crop_generator(val_generator, 32)
        ts_datagen = ImageDataGenerator(
                    )  

        ts_datagen.fit(x_test)  
        test_generator = ts_datagen.flow(x_test, y_test, batch_size=batch_size)        

        test_crops = crop_generator_center(test_generator, 32)
    else:
        from keras.callbacks import LearningRateScheduler

        epochs = 700
        episodes  = 1
        tr_datagen = ImageDataGenerator(
                        horizontal_flip=True,
                        validation_split=1-0.9,
                        )
        tr_datagen.fit(x_train)  
        train_generator = tr_datagen.flow(x_train, y_train, batch_size=batch_size)
        train_crops = crop_generator(train_generator, 32)

        ts_datagen = ImageDataGenerator(
                           )  

        ts_datagen.fit(x_test)  
        test_generator = ts_datagen.flow(x_test, y_test, batch_size=batch_size)        

        test_crops = crop_generator_center(test_generator, 32)    


            
            
    # kernel_1, filters_1, skip_connect
    total_val_acc = []
    total_val_loss = []


    total_reward = []
    total_test_acc = []
    total_test_loss = []


    total_action = []
    total_code = []


    for i in range(episodes):  
        #hyperparameters first action 
        if episodesCount ==0:
                if search:
                    action_in = np.random.choice(range(1,30), size=origin_len).astype(int)
                    print("===Generating random action to explore===")      
                else:
                    with open ('../scr/cifar10/result/best_action', 'rb') as fp:
                        action_in = pickle.load(fp)    
                    print("===Come back to best action===")                 
        elif episodesCount > 20:
            action_in = np.random.choice(range(1,30), size=origin_len).astype(int)
            print("===Generating random action to explore===")
        elif np.random.random() < 0.1 :            
            action_in = np.random.choice(range(1,30), size=origin_len).astype(int)
            print("===Generating random action to explore===")
        elif np.random.random() > 0.9:
            with open ('../scr/cifar10/result/best_action', 'rb') as fp:
                action_in = pickle.load(fp)    
            print("===Come back to best action===")


        print("Parameter:",np.reshape(action_in,[-1,4]))



        #====================state====================
        total_action.append(np.reshape(action_in,[1,-1]))


        DNNcoder = coder() 
        DNNcoder.restore()
   
        from controller import controller  
        print('==========',episodesCount,'===========')
        if episodesCount % 30 == 0:
            pre_init = DNNcoder.show_weight(np.reshape(action_in,[1,-1]))
            controller = controller(pre_init, origin_len=origin_len, embedding=embedding)
            episodesCount = 29
            
        else:
            pre_init = DNNcoder.show_weight(np.reshape(action_in,[1,-1]))
            controller = controller(pre_init, origin_len=origin_len, embedding=embedding)                   
            controller.restore() 
            episodesCount -=  1


        #====================action====================
        code = controller.preds(state = np.reshape(action_in,[1,-1]))
        total_code.append(code)


        if episodesCount < 20:
            print('==========Model Training...==========')
            re_total_action = np.reshape(total_action,[-1,origin_len])
            re_total_code = np.reshape(total_code,[-1,embedding])
            for w in range(int(i/10)):
                if w == 0:
                    shuffer_ind = random.sample(range(re_total_reward.shape[0]),3)
                    shuffer_ind.append(-1)
                else:
                    shuffer_ind = random.sample(range(re_total_reward.shape[0]),4)
                controller.train(state = re_total_action[shuffer_ind], reward = re_total_reward[shuffer_ind], action = re_total_code[shuffer_ind])
            loss_control = controller.loss(state = re_total_action[shuffer_ind], reward = re_total_reward[shuffer_ind], action = re_total_code[shuffer_ind])

        else:
            loss_control = 0


        controller.save()
        code = controller.preds(state = np.reshape(action_in,[1,-1]))
        print(code)

            
            
        #check action > 0 and next action
        if i ==0:
            if search:
                next_action_in = action_in
            else: 
                next_action_in = action_in[0]
        else:    
            next_action_in = np.ceil(np.reshape(DNNcoder.pred(fake,code,-1),[-1])).astype(int)

        k=0             
        for a in next_action_in:
            if a <= 0:
                next_action_in[k] = 1
            k+=1


        action_in = np.reshape(next_action_in,[-1,4])    


        del DNNcoder, controller
        gc.collect()
        K.clear_session()             
        tf.set_random_seed(1)

        model = model_fn(action_in, addFilter)

        if search:  
            sgd = keras.optimizers.SGD(lr=0.05, momentum=0.9, nesterov=True,decay=0)

            model.compile(sgd, 'categorical_crossentropy', metrics=['accuracy'])    
            mcp = keras.callbacks.ModelCheckpoint("../scr/cifar10/finalModel/best_weights.hdf5", monitor="val_acc",
                              save_best_only=True, save_weights_only=False)        


            schedule = SGDRScheduler(min_lr=0.001,                                    
                                 max_lr=0.05,
                                 steps_per_epoch=50000*train_rate//batch_size,
                                 cycle_length=10,
                                 mult_factor=2)
            history = model.fit_generator(train_crops,
                                    steps_per_epoch= 50000*train_rate // batch_size,
                                    validation_data = val_crops,
                                    epochs=epochs,
                                    shuffle=True,
                                    validation_steps= 50000*(1-train_rate)//batch_size,
                                    callbacks=[schedule,mcp])
            val_acc = max(history.history['val_acc'])
            tr_acc = max(history.history['acc'])


           #load best model
            model.load_weights("../scr/cifar10/finalModel/best_weights.hdf5")
#             reward =  (val_acc  -  (tr_acc - val_acc))**3
            reward =  val_acc**3 

            total_val_acc.append(val_acc)
            total_reward.append(reward)
            nor_total_reward = preprocessing.scale(total_reward)
            re_total_reward = np.reshape(nor_total_reward,[-1,1])  
        else:                
            schedule = SGDRScheduler(min_lr=0.001,                                    
                                 max_lr=0.05,
                                 steps_per_epoch=50000//batch_size,
                                 cycle_length=epochs,
                                 mult_factor=1)
            sgd = keras.optimizers.SGD(lr=0.05,momentum=0.9, nesterov=True,decay=0)
            model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])    
            mcp = keras.callbacks.ModelCheckpoint("../scr/cifar10/finalModel/best_weights.hdf5", monitor="val_acc",
                              save_best_only=True, save_weights_only=False)        
            history = model.fit_generator(train_crops,
                                    steps_per_epoch=50000// batch_size,
                                    validation_data = test_crops,
                                    epochs=epochs,
                                    shuffle=True,
                                    validation_steps= 10000//batch_size,
                                    callbacks=[mcp,MyCallback(),schedule])   
            val_acc = max(history.history['val_acc'])
            model.load_weights("../scr/cifar10/finalModel/best_weights.hdf5")


            #
        test_acc = model.evaluate_generator(test_crops, 10000 / batch_size)[1]
        loss = model.evaluate_generator(test_crops, 10000 / batch_size)[0]
        total_test_acc.append(test_acc)

        if search:
            if val_acc == max(total_val_acc):
               plot_model(model, to_file='../scr/cifar10/result/model_graph.png')
               with open('../scr/cifar10/result/modelsummary.txt', 'w') as f:
                    with redirect_stdout(f):
                         model.summary()

               with open('../scr/cifar10/result/best_acc.txt', 'w') as f:
                    with redirect_stdout(f):
                         print(i,"val_acc:",val_acc,"test_acc:",test_acc)

               with open('../scr/cifar10/result/best_action', 'wb') as fp:
                   pickle.dump(np.reshape(action_in,[1,-1]), fp)

               print("=====================================Model Save - Best====================================")  


            print(i,"accuracy_val:",val_acc,"accuracy_test:",test_acc,"reward:",re_total_reward[-1],'model_loss:',loss,"control_loss:",loss_control)


            plt.plot(total_val_acc, label='VAL')
            plt.plot(total_test_acc, label='TEST')
            plt.legend(loc='lower left')
            plt.savefig('../scr/cifar10/result/training_history.png')

            del history
            gc.collect()
            K.clear_session() 
            plt.close('all')

        else:
            plot_model(model, to_file='model_graph.png')
            with open('../scr/cifar10/result/modelsummary.txt', 'w') as f:
                 with redirect_stdout(f):
                      model.summary()

            with open('../scr/cifar10/result/best_acc.txt', 'w') as f:
                 with redirect_stdout(f):
                      print("test_acc:",val_acc,test_acc)

            sys.exit()        

if __name__ == '__main__':
    main()