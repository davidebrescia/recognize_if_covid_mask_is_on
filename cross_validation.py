
# Cross-validation

# Here it is shown a cross-validation approach that has been implemented during the homework to see  
# if its general goodness could be useful for our task.
# Only the its implementation has been reported to not make it so long with all the previous lines of code 
# which are the same of the Experiment 1 notebook. 

# KFold Object for dividing the dataset at each iteration
from sklearn.model_selection import KFold

# This specifies the number of folds in which the training set is divided: 10 has been chosen because 
# in general it is the best solution for cross-validation
num_folds = 10

# This quantity specifies the lenght of the dataset
n_samples = len(dataset)

# Instance of KFold object 
kfold = KFold(n_splits=num_folds, shuffle=True)

# for cycle in which we divide the training set, then train the model with that split
for train_index, val_index in kfold.split(np.zeros(n_samples), data["class"]):
  train_dataframe = data.iloc[train_index]
  valid_dataframe = data.iloc[val_index]
  



  # Training
  training_dir = os.path.join(dataset_dir, 'training')
  train_gen = train_data_gen.flow_from_dataframe(train_dataframe,
                                                training_dir,
                                                x_col="filename",
                                                y_col="class",
                                                batch_size=bs,
                                                classes=classes,
                                                class_mode='categorical',
                                                target_size=(img_h,img_w),
                                                shuffle=True,
                                                seed=SEED)
  
  # Validation
  valid_gen = valid_data_gen.flow_from_dataframe(valid_dataframe,
                                                training_dir,
                                                x_col="filename",
                                                y_col="class",
                                                batch_size=bs,
                                                classes=classes,
                                                class_mode='categorical',
                                                target_size=(img_h,img_w),
                                                shuffle=True,
                                                seed=SEED)
  

  train_dataset = tf.data.Dataset.from_generator(lambda: train_gen,
                                               output_types=(tf.float32, tf.float32),
                                               output_shapes=([None, img_h, img_w, 3], [None, num_classes]))
  
  valid_dataset = tf.data.Dataset.from_generator(lambda: valid_gen, 
                                               output_types=(tf.float32, tf.float32),
                                               output_shapes=([None, img_h, img_w, 3], [None, num_classes]))
  valid_dataset = valid_dataset.repeat()
  train_dataset = train_dataset.repeat()
  

  # Model Train

  # for model training different number of epochs have been tried, but at the end 25 epochs led us to better results than 
  # the ones provided with smaller number of epochs because the train was stopped before complete.
  STEP_SIZE_VALID=valid_gen.n//valid_gen.batch_size
  STEP_SIZE_TRAIN=train_gen.n//train_gen.batch_size
  model.fit(x=train_dataset,
            epochs=25,
            steps_per_epoch=STEP_SIZE_TRAIN,
            validation_data=valid_dataset,
            validation_steps=STEP_SIZE_VALID,
            callbacks=callbacks)