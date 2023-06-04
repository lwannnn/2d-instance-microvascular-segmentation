from Models.exampleModel import UNet
def train_model(model_name):
    print("model_name:"+str(model_name))
    if model_name == 'test':
        return UNet()


if __name__=='__main__':
    print(train_model('test'))