import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
import seaborn as sns


train_data= open('D:/Dropbox/나메렝/coursework/2022 2학기/통계적패턴인식/Hw1/train.txt', 'r')
test_data= open('D:/Dropbox/나메렝/coursework/2022 2학기/통계적패턴인식/Hw1/test.txt', 'r')


# dataset to numpy
np_traindata=np.loadtxt('D:/Dropbox/나메렝/coursework/2022 2학기/통계적패턴인식/Hw1/train.txt', dtype = 'str')
np_testdata=np.loadtxt('D:/Dropbox/나메렝/coursework/2022 2학기/통계적패턴인식/Hw1/test.txt', dtype = 'str')

np_train_data=np_traindata.astype(float)
np_test_data=np_testdata.astype(float)

print('Train data shape: ', np.shape(np_train_data))
print('Test data shape: ', np.shape(np_test_data))


def scatter(np_train_data, np_test_data):
    plt.scatter(np_train_data[:,0][:600], np_train_data[:,1][:600], s=2, label='$y=0$') #class0
    plt.scatter(np_train_data[:,0][600:], np_train_data[:,1][600:], s=2, label='$y=1$') #class1
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.title('training data')
    plt.legend()
    plt.show()

    plt.scatter(np_test_data[:,0],np_test_data[:,1], s=2)
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.title('test data')
    plt.show()



def distance(a, b):
    # Euclidean distance

    temp = a - b
    distance = np.sqrt(np.dot(temp.T, temp))

    return distance


def distances(num_traindata, np_train_data):
    # Calculate distances
    distances = np.zeros((num_traindata, num_traindata))
    for i in range(num_traindata):
        for j in range(num_traindata):
            if i == j:
                distances[i][j] = 10000
            elif distances[j][i] != 0:
                distances[i][j] = distances[j][i]
            else:
                distances[i][j] = distance(np_train_data[i][:2], np_train_data[j][:2])
    return distances


# Values
num_traindata = 1000
num_testdata = 3000
num_class0 = 600
num_class1 = num_traindata - num_class0


def plotLog(log):  # weight가 업데이트되어가는 로그
    df = pd.DataFrame(log)
    df.columns = ['x', 'y']
    df['z'] = np.array(list(range(len(df))))
    #  df['z'] = np.log(df['z']+1)
    sns.scatterplot(data=df, x='x', y='y', hue='z')


def adam(gfx, x, theta, ir=2, alpha=0.1, beta1=0.9, beta2=0.999, epsilon=10e-8, th=0.00001):  # alpha= learning_rate
    m = 0
    v = 0
    t = 1
    log = np.array([])
    while t < ir:
        log = np.append(log, theta)
        gx = gfx(x, theta)
        m = beta1 * m + (1 - beta1) * gx
        v = beta2 * v + (1 - beta2) * gx ** 2
        mh = m / (1 - beta1 ** t)  # m hat
        vh = v / (1 - beta2 ** t)  # v hat
        theta_new = theta - alpha * mh / (vh ** (1 / 2) + epsilon)

        if (sum(abs(theta - theta_new)) < th):
            break
        theta = theta_new
        t += 1
    log = log.reshape(len(log) // 2, 2)
    return theta, t, log


def fx(x, theta):
    theta1, theta2 = theta
    return -(1 / 2) * np.log(2 * np.pi * theta2) - (1 / (2 * theta2)) * ((x - theta1) ** 2)


def gfx(x, theta):
    theta1, theta2 = theta
    gradient_theta1 = (1 / theta2) * (x - theta1)
    gradient_theta2 = -(1 / (2 * theta2)) + ((x - theta1) ** 2) / (2 * theta2 ** 2)
    return np.array([gradient_theta1, gradient_theta2])





def fx(x, theta):
    theta1, theta2 = theta
    return -(1 / 2) * np.log(2 * np.pi * theta2) - (1 / (2 * theta2)) * ((x - theta1) ** 2)


def gfx(x, theta):
    theta1, theta2 = theta
    gradient_theta1 = (1 / theta2) * (x - theta1)
    gradient_theta2 = -(1 / (2 * theta2)) + ((x - theta1) ** 2) / (2 * theta2 ** 2)
    return np.array([gradient_theta1, gradient_theta2])



class0_feature1_theta = [20, 3]
class0_feature2_theta = [20, 3]
class1_feature1_theta = [20, 3]
class1_feature2_theta = [20, 3]

class0_feature1_theta_log = []
class0_feature2_theta_log = []
class1_feature1_theta_log = []
class1_feature2_theta_log = []

epoch = 500
learning_rate = 0.01




for e in range(epoch):

    # Train
    for i in range(num_traindata):
        # class 0
        class0_feature1_theta, _, log_1 = adam(gfx, np_train_data[i][0], class0_feature1_theta, alpha=learning_rate)
        class0_feature2_theta, _, log_2 = adam(gfx, np_train_data[i][1], class0_feature2_theta, alpha=learning_rate)
        # class 1
        class1_feature1_theta, _, log_3 = adam(gfx, np_train_data[i][0], class1_feature1_theta, alpha=learning_rate)
        class1_feature2_theta, _, log_4 = adam(gfx, np_train_data[i][1], class1_feature2_theta, alpha=learning_rate)

        if i % 1000 == 0:
            class0_feature1_theta_log.append(log_1)
            class0_feature2_theta_log.append(log_2)
            class1_feature1_theta_log.append(log_3)
            class1_feature2_theta_log.append(log_4)

    # Test with training data itself
    predicts = np.zeros((num_traindata))
    for i in range(num_traindata):
        # class0
        log_likelihood_class0_feature1 = fx(np_train_data[i][0], class0_feature1_theta)
        log_likelihood_class0_feature2 = fx(np_train_data[i][1], class0_feature2_theta)
        log_likelihood_class0 = log_likelihood_class0_feature1 + log_likelihood_class0_feature2
        # class1
        log_likelihood_class1_feature1 = fx(np_train_data[i][0], class1_feature1_theta)
        log_likelihood_class1_feature2 = fx(np_train_data[i][1], class1_feature2_theta)
        log_likelihood_class1 = log_likelihood_class1_feature1 + log_likelihood_class1_feature2

        if log_likelihood_class0 > log_likelihood_class1:
            predicts[i] = 0
        elif log_likelihood_class0 < log_likelihood_class1:
            predicts[i] = 1
        else:
            print('same posterior')
            break

    # errors
    errors = (predicts != np_train_data[:, 2]).sum()
    if e%10==0:
        print('Epoch: ', e)
        print('Number of errors: ', errors, ' /', num_traindata)
        print('Accuracy: ', 100 * ((num_traindata - errors) / num_traindata), '%')

    if e == epoch - 1:
        for log_idx in range(len(class0_feature1_theta_log)):
            plotLog(class0_feature1_theta_log[log_idx])
        plt.title('class0_feature1_theta')
        plt.legend('', frameon=False)
        plt.show()

        for log_idx in range(len(class0_feature2_theta_log)):
            plotLog(class0_feature2_theta_log[log_idx])
        plt.title('class0_feature2_theta')
        plt.legend('', frameon=False)
        plt.show()

        for log_idx in range(len(class1_feature1_theta_log)):
            plotLog(class1_feature1_theta_log[log_idx])
        plt.title('class1_feature1_theta')
        plt.legend('', frameon=False)
        plt.show()

        for log_idx in range(len(class1_feature2_theta_log)):
            plotLog(class1_feature2_theta_log[log_idx])
        plt.title('class1_feature2_theta')
        plt.legend('', frameon=False)
        plt.show()

# 초기값 다시 잡고
# beta 조절해서 크게 바뀌는지 확ㅜㅎ ㅎ인




