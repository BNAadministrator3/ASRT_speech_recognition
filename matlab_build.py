import os
import re
import matplotlib.pyplot as plt



Test_Report_loss = open(os.path.join(os.getcwd(),'speech_log_file','Test_Report_loss.txt'),encoding='utf8')
Test_Report_accuracy = open(os.path.join(os.getcwd(),'speech_log_file','Test_Report_accuracy.txt'),encoding='utf8')

print('开始读取')
if False:
    loss_list_y = []
    loss_list_x = []
    for i,line in enumerate(Test_Report_loss):
        # if i %3 == 0:
        strs = line.split(':')
        s = strs[-1][:-1]
        if float(s) >75:
            print(line)
        loss_list_y.append(float(s))
        loss_list_x.append(i)
    print('画图')
    plt.plot(loss_list_x,loss_list_y)
    plt.title('speech model loss curve')
    plt.xlabel('step')
    plt.ylabel('loss')
    for i in range(1,len(loss_list_x)+1):
        if i % 10000 == 0:
            plt.text(loss_list_x[i],loss_list_y[i],str(round(loss_list_y[i],2)),ha='center',va='bottom',fontsize=10.5)
    plt.savefig(os.path.join(os.getcwd(),'speech_log_file','Test_Report_loss.jpg'))
    plt.show()
else:
    accuracy_list_y = []
    accuracy_list_x = []
    accuracy_num = 0
    for i,line in enumerate(Test_Report_accuracy):
        strs = line.find('错误率')
        if strs >= 0:
            strs = re.findall(r'\d+\.\d+',line)
            s = round(float(strs[-1]),4)
            accuracy_list_y.append(s)
            accuracy_list_x.append(accuracy_num)
            accuracy_num +=1

    plt.plot(accuracy_list_x,accuracy_list_y)
    plt.title('acoustic model error rate curve')
    plt.xlabel('epoch')
    plt.ylabel('Statement error rate')
    for i in range(1,len(accuracy_list_x)+1):
        if i % 10 == 0:
            plt.text(accuracy_list_x[i],accuracy_list_y[i],str(round(accuracy_list_y[i],2)),ha='center',va='bottom',fontsize=9)
    plt.savefig(os.path.join(os.getcwd(),'speech_log_file','Test_Report_accuracy.jpg'))
    plt.show()