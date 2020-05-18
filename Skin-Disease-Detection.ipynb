{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Importing the Libraries"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Using TensorFlow backend.\n"
     ]
    }
   ],
   "source": [
    "from keras.models import Sequential\n",
    "from keras.layers import Dense\n",
    "from keras.layers import Convolution2D\n",
    "from keras.layers import MaxPooling2D\n",
    "from keras.layers import Flatten"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "WARNING:tensorflow:From C:\\Users\\Admin\\Anaconda3\\lib\\site-packages\\keras\\backend\\tensorflow_backend.py:74: The name tf.get_default_graph is deprecated. Please use tf.compat.v1.get_default_graph instead.\n",
      "\n"
     ]
    }
   ],
   "source": [
    "#Intialize the model\n",
    "model=Sequential()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "WARNING:tensorflow:From C:\\Users\\Admin\\Anaconda3\\lib\\site-packages\\keras\\backend\\tensorflow_backend.py:517: The name tf.placeholder is deprecated. Please use tf.compat.v1.placeholder instead.\n",
      "\n",
      "WARNING:tensorflow:From C:\\Users\\Admin\\Anaconda3\\lib\\site-packages\\keras\\backend\\tensorflow_backend.py:4138: The name tf.random_uniform is deprecated. Please use tf.random.uniform instead.\n",
      "\n"
     ]
    }
   ],
   "source": [
    "# Add Convolution Layer\n",
    "model.add(Convolution2D(32,(3,3),input_shape=(64,64,3),activation=\"relu\"))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "WARNING:tensorflow:From C:\\Users\\Admin\\Anaconda3\\lib\\site-packages\\keras\\backend\\tensorflow_backend.py:3976: The name tf.nn.max_pool is deprecated. Please use tf.nn.max_pool2d instead.\n",
      "\n"
     ]
    }
   ],
   "source": [
    "#Add Pooling Layer\n",
    "model.add(MaxPooling2D(pool_size = (2, 2)))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Add Flattening Layer\n",
    "model.add(Flatten())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\Admin\\Anaconda3\\lib\\site-packages\\ipykernel_launcher.py:2: UserWarning: Update your `Dense` call to the Keras 2 API: `Dense(activation=\"relu\", units=120, kernel_initializer=\"random_uniform\")`\n",
      "  \n"
     ]
    }
   ],
   "source": [
    "#Add Hidden Layer\n",
    "model.add(Dense(init=\"random_uniform\",activation=\"relu\",output_dim=120))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\Admin\\Anaconda3\\lib\\site-packages\\ipykernel_launcher.py:2: UserWarning: Update your `Dense` call to the Keras 2 API: `Dense(activation=\"sigmoid\", units=1, kernel_initializer=\"random_uniform\")`\n",
      "  \n"
     ]
    }
   ],
   "source": [
    "#Add Output layer\n",
    "model.add(Dense(init=\"random_uniform\",activation=\"sigmoid\",output_dim=1))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Data preprocessing"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [],
   "source": [
    "from keras.preprocessing.image import ImageDataGenerator\n",
    "\n",
    "train_datagen = ImageDataGenerator(rescale = 1./255,\n",
    "                                   shear_range = 0.2,\n",
    "                                   zoom_range = 0.2,\n",
    "                                   horizontal_flip = True)\n",
    "\n",
    "test_datagen = ImageDataGenerator(rescale = 1./255)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Found 77 images belonging to 2 classes.\n",
      "Found 30 images belonging to 2 classes.\n"
     ]
    }
   ],
   "source": [
    "x_train = train_datagen.flow_from_directory('C:/Users/Admin/OneDrive/Desktop/Dataset/TrainData',\n",
    "                                                 target_size = (64, 64),\n",
    "                                                 batch_size = 32,\n",
    "                                                     class_mode = 'binary')\n",
    "x_test = test_datagen.flow_from_directory('C:/Users/Admin/OneDrive/Desktop/Dataset/TestData',\n",
    "                                            target_size = (64, 64),\n",
    "                                            batch_size = 32,\n",
    "                                            class_mode = 'binary')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "{'Psoriasis': 0, 'SkinCancer': 1}\n"
     ]
    }
   ],
   "source": [
    "print(x_train.class_indices)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "WARNING:tensorflow:From C:\\Users\\Admin\\Anaconda3\\lib\\site-packages\\keras\\optimizers.py:790: The name tf.train.Optimizer is deprecated. Please use tf.compat.v1.train.Optimizer instead.\n",
      "\n",
      "WARNING:tensorflow:From C:\\Users\\Admin\\Anaconda3\\lib\\site-packages\\keras\\backend\\tensorflow_backend.py:3376: The name tf.log is deprecated. Please use tf.math.log instead.\n",
      "\n",
      "WARNING:tensorflow:From C:\\Users\\Admin\\Anaconda3\\lib\\site-packages\\tensorflow\\python\\ops\\nn_impl.py:180: add_dispatch_support.<locals>.wrapper (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.\n",
      "Instructions for updating:\n",
      "Use tf.where in 2.0, which has the same broadcast rule as np.where\n"
     ]
    }
   ],
   "source": [
    "#Compile the model\n",
    "model.compile(loss=\"binary_crossentropy\",optimizer=\"adam\",metrics=[\"accuracy\"])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "WARNING:tensorflow:From C:\\Users\\Admin\\Anaconda3\\lib\\site-packages\\keras\\backend\\tensorflow_backend.py:986: The name tf.assign_add is deprecated. Please use tf.compat.v1.assign_add instead.\n",
      "\n",
      "Epoch 1/10\n",
      "200/200 [==============================] - 74s 369ms/step - loss: 0.0031 - acc: 1.0000 - val_loss: 1.1921e-07 - val_acc: 1.0000\n",
      "Epoch 2/10\n",
      "200/200 [==============================] - 73s 365ms/step - loss: 1.1921e-07 - acc: 1.0000 - val_loss: 1.1921e-07 - val_acc: 1.0000\n",
      "Epoch 3/10\n",
      "200/200 [==============================] - 78s 392ms/step - loss: 1.1921e-07 - acc: 1.0000 - val_loss: 1.1921e-07 - val_acc: 1.0000\n",
      "Epoch 4/10\n",
      "200/200 [==============================] - 78s 388ms/step - loss: 1.1921e-07 - acc: 1.0000 - val_loss: 1.1921e-07 - val_acc: 1.0000\n",
      "Epoch 5/10\n",
      "200/200 [==============================] - 76s 382ms/step - loss: 1.1921e-07 - acc: 1.0000 - val_loss: 1.1921e-07 - val_acc: 1.0000\n",
      "Epoch 6/10\n",
      "200/200 [==============================] - 75s 376ms/step - loss: 1.1921e-07 - acc: 1.0000 - val_loss: 1.1921e-07 - val_acc: 1.0000\n",
      "Epoch 7/10\n",
      "200/200 [==============================] - 75s 376ms/step - loss: 1.1921e-07 - acc: 1.0000 - val_loss: 1.1921e-07 - val_acc: 1.0000\n",
      "Epoch 8/10\n",
      "200/200 [==============================] - 75s 373ms/step - loss: 1.1921e-07 - acc: 1.0000 - val_loss: 1.1921e-07 - val_acc: 1.0000\n",
      "Epoch 9/10\n",
      "200/200 [==============================] - 75s 374ms/step - loss: 1.1921e-07 - acc: 1.0000 - val_loss: 1.1921e-07 - val_acc: 1.0000\n",
      "Epoch 10/10\n",
      "200/200 [==============================] - 75s 374ms/step - loss: 1.1921e-07 - acc: 1.0000 - val_loss: 1.1921e-07 - val_acc: 1.0000\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<keras.callbacks.History at 0x1b99c224908>"
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model.fit_generator(x_train,\n",
    "                         steps_per_epoch = 200,\n",
    "                         epochs = 10,\n",
    "                         validation_data = x_test,\n",
    "                         validation_steps = 63)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [],
   "source": [
    "model.save(\"cnnmodel.h5\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# configuring the learning process"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Compile the model\n",
    "model.compile(loss=\"binary_crossentropy\",optimizer=\"adam\",metrics=[\"accuracy\"])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Train the model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/10\n",
      "200/200 [==============================] - 75s 377ms/step - loss: 1.1921e-07 - acc: 1.0000 - val_loss: 1.1921e-07 - val_acc: 1.0000\n",
      "Epoch 2/10\n",
      "200/200 [==============================] - 73s 365ms/step - loss: 1.1921e-07 - acc: 1.0000 - val_loss: 1.1921e-07 - val_acc: 1.0000\n",
      "Epoch 3/10\n",
      "200/200 [==============================] - 70s 348ms/step - loss: 1.1921e-07 - acc: 1.0000 - val_loss: 1.1921e-07 - val_acc: 1.0000\n",
      "Epoch 4/10\n",
      "200/200 [==============================] - 69s 346ms/step - loss: 1.1921e-07 - acc: 1.0000 - val_loss: 1.1921e-07 - val_acc: 1.0000\n",
      "Epoch 5/10\n",
      "200/200 [==============================] - 71s 353ms/step - loss: 1.1921e-07 - acc: 1.0000 - val_loss: 1.1921e-07 - val_acc: 1.0000\n",
      "Epoch 6/10\n",
      "200/200 [==============================] - 70s 350ms/step - loss: 1.1921e-07 - acc: 1.0000 - val_loss: 1.1921e-07 - val_acc: 1.0000\n",
      "Epoch 7/10\n",
      "200/200 [==============================] - 75s 374ms/step - loss: 1.1921e-07 - acc: 1.0000 - val_loss: 1.1921e-07 - val_acc: 1.0000\n",
      "Epoch 8/10\n",
      "200/200 [==============================] - 74s 370ms/step - loss: 1.1921e-07 - acc: 1.0000 - val_loss: 1.1921e-07 - val_acc: 1.0000\n",
      "Epoch 9/10\n",
      "200/200 [==============================] - 73s 367ms/step - loss: 1.1921e-07 - acc: 1.0000 - val_loss: 1.1921e-07 - val_acc: 1.0000\n",
      "Epoch 10/10\n",
      "200/200 [==============================] - 74s 371ms/step - loss: 1.1921e-07 - acc: 1.0000 - val_loss: 1.1921e-07 - val_acc: 1.0000\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<keras.callbacks.History at 0x1b99c4cb408>"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model.fit_generator(x_train,\n",
    "                         steps_per_epoch = 200,\n",
    "                         epochs = 10,\n",
    "                         validation_data = x_test,\n",
    "                         validation_steps = 63)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# save the model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [],
   "source": [
    "model.save(\"cnnmodel.h5\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Requirement already satisfied: opencv-contrib-python in c:\\users\\admin\\anaconda3\\lib\\site-packages (4.2.0.34)\n",
      "Requirement already satisfied: numpy>=1.14.5 in c:\\users\\admin\\anaconda3\\lib\\site-packages (from opencv-contrib-python) (1.16.5)\n"
     ]
    }
   ],
   "source": [
    "!pip install opencv-contrib-python"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# predicting the model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [],
   "source": [
    "from keras.models import load_model\n",
    "import numpy as np\n",
    "import cv2\n",
    "model=load_model('cnnmodel.h5')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [],
   "source": [
    "model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [],
   "source": [
    "from skimage.transform import resize\n",
    "\n",
    "def detect(frame):\n",
    "    try:\n",
    "        img = resize(frame,(64,64))\n",
    "        img = np.expand_dims(img,axis=0)\n",
    "        if(np.max(img)>1):\n",
    "            img = img/255.0\n",
    "        prediction = model.predict(img)\n",
    "        print(prediction)\n",
    "        prediction = model.predict_classes(img)\n",
    "        print(prediction)\n",
    "    except AttributeError:\n",
    "        print(\"shape not found\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[1.]]\n",
      "[[1]]\n"
     ]
    }
   ],
   "source": [
    "frame=cv2.imread(\"C:/Users/Admin/OneDrive/Desktop/Dataset/TrainData/Psoriasis/download.jfif\")\n",
    "data = detect(frame)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
