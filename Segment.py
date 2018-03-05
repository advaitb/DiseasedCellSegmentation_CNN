class Segment(object):

    # path = ""

    def __init__(self, data_dir=""):
        """
            data_directory : path like /home/rajat/nnproj/dataset/
                            includes the dataset folder with '/'
            Initialize all your variables here
        """
        self.model = Sequential()

        self.model.add(Conv2D(4, (3, 3), input_shape=(None, None, 3), padding='same', activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2,2)))

        self.model.add(Conv2D(8, (3, 3), padding='same', activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2,2)))

        self.model.add(Conv2D(16, (3, 3), padding='same', activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2,2)))

        self.model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))

        self.model.add(UpSampling2D(size=(2,2)))
        self.model.add(Conv2DTranspose(16, (3,3), activation='relu', padding='same'))

        self.model.add(UpSampling2D(size=(2,2)))
        self.model.add(Conv2DTranspose(8, (3,3), activation='relu', padding='same'))

        self.model.add(UpSampling2D(size=(2,2)))
        self.model.add(Conv2DTranspose(4, (3,3), activation='relu', padding='same'))

        self.model.add(Conv2D(1, (1, 1), padding='same', activation='sigmoid'))

        self.model.compile(loss='binary_crossentropy',
                      optimizer='Adagrad',
                      metrics=[IoUscore])


    def train(self):
        """
            Trains the model on data given in path/train.csv

            No return expected
        """
        list_id = [i for i in range(164)]

        train = []
        for id in list_id:
            input_file = 'Train_Data/train-'+str(id)+'.jpg'
            input_img = cv2.imread(input_file)
            train.append(input_img)

        train = np.array(train)
        train = train.reshape((train.shape[0],train.shape[1],train.shape[2],3))
        train_set = train/255.0
        #scaled and separated

        train = []
        for id in list_id:
            input_file = 'Train_Data/train-'+str(id)+'-mask.jpg'
            input_img = cv2.imread(input_file, flags=0)
            train.append(input_img)

        train = np.array(train)
        train = train.reshape((train.shape[0],train.shape[1],train.shape[2],1))
        label_train = train/255.0

        self.model.fit(train_set, label_train,  epochs=10, batch_size=15)

    def get_mask(self, image):
        """
            image : a variable resolution RGB image in the form of a numpy array

            return: A list of lists with the same 2d size as of the input image with either 0 or 1 as each entry

        """
        image_in = image.reshape((1,) + image.shape)
        predicted = np.round(self.model.predict(image_in))
        # print image_in.shape, predicted.shape
        predicted1 = predicted.reshape((predicted.shape[1],predicted.shape[2])).astype('uint64')
        if image.shape[1] != predicted1.shape[1]:
            predicted1 = np.hstack((predicted1,np.zeros((predicted1.shape[0],image.shape[1]-predicted1.shape[1]))))
        if image.shape[0] != predicted1.shape[0]:
            predicted1 = np.vstack((predicted1,np.zeros((image.shape[0]-predicted1.shape[0],predicted1.shape[1]))))
        # cv2.imwrite('testing.png',predicted1*255)
        return predicted1.astype('uint64')
