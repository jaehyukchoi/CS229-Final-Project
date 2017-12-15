import ast
import os
import numpy as np
import matplotlib.pyplot as plt
import itertools

# Function taken from http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    epsilon = 1e-10
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    cm = np.rot90(cm)
    cm = np.rot90(cm)
    cm = np.transpose(cm)
    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

loss = np.load('loss.npy')

plt.plot(loss)
plt.xlabel('Epochs')
plt.ylabel('Cross-Entropy Loss')
plt.title('Loss of Softmax Regression')
plt.savefig('loss')
# plt.show()

train_acc = np.load('train_acc.npy')
valid_acc = np.load('valid_acc.npy')

plt.figure()
plt.plot(train_acc)
plt.plot(valid_acc)
plt.xlabel('Epochs')
plt.ylabel('Training Accuracy')
plt.title('Softmax Regression Accuracy vs Epoch')
plt.legend(['train acc','valid acc'])
plt.savefig('accuracy')
# plt.show()

plt.figure()
cm = np.load('test_confusion.npy')
class_names = [0,1,2,3,4,5,6,7,8,9]
plot_confusion_matrix(cm, classes=class_names, normalize=False,
                      title='Test Confusion Matrix for Softmax Regression')
plt.savefig('test_confusion')
plt.show()


