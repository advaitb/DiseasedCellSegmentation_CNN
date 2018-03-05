#Save and Load Model for continous training

def save_model(self, **params):

        # file_name = params['name']
        # pickle.dump(self, gzip.open(file_name, 'wb'))

        """
            saves model on the disk

            no return expected
        """

        json_string = self.model.to_json()
        with open('model.config', 'w') as outfile:
            json.dump(json_string, outfile)
        self.model.save_weights('my_model_weights.h5')

    @staticmethod
def load_model(**params):

        # file_name = params['name']
        # return pickle.load(gzip.open(file_name, 'rb'))

        """
            returns a pre-trained instance of Segment class
        """
     with open('model.config') as in_file:
        json_string = json.load(in_file)
        model = model_from_json(json_string)
        model.load_weights('my_model_weights.h5')
        model.compile(loss='mean_squared_error',
                      optimizer='adagrad',
                      metrics=[IoUscore])
        
        obj = Segment()
        obj.model = model
        return obj
