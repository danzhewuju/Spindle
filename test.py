import threading
import time

exitFlag = 0


class myThread(threading.Thread):  # 继承父类threading.Thread
    def __init__(self, threadID, name, a):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.a = a

    def run(self):  # 把要执行的代码写到run函数里面 线程在创建后会直接运行run函数
        self.a.append(12)


# 创建新线程
a = []
thread1 = myThread(1, "Thread-1", a)
thread2 = myThread(2, "Thread-2", a)

# 开启线程
thread1.start()
thread2.start()

print(a)