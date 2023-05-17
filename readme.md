### Data augmentation
I have successfully applied the augmentation, though the accuracy may not be improved significantly, while the loss and the training accuracy do not increase that high than before. Therefore, I think it prevents the overfitting here, which is kind of an improvement. 

Also, I have tried to train for some longer epochs, since overfitting is less significant. However, it takes a little bit longer time, I cannot make some conclusions by now. In my opinion, most of the training time is wasted on data transfering or copying. Maybe we could apply a deeper and more recent model to get more accuracy and make more use of good GPUs, since my GPU did not even sweat during training. 

Once we combine the data augmentation with the pseudolabel, maybe we can get a much better result.