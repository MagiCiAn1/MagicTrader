import importlib
import inspect
import os

import schedule


def singleton(cls):
    instances = {}

    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return get_instance


@singleton
class Task(object):
    tasks = []
    context = {}

    def add_run_task(self, job, run_time):
        caller_frame = inspect.currentframe().f_back
        caller_file = inspect.getframeinfo(caller_frame).filename
        file_name = os.path.basename(caller_file)
        t = (job, run_time, file_name)
        self.tasks.append(t)
        print(self.tasks)

    def run_task(self):
        print("Context:", self.context)
        print("Tasks:", self.tasks)
        for item in self.tasks:
            print(item)
            key = item[2]
            g = self.context[key]
            schedule.every().day.at(item[1]).do(item[0], g)

    def init_strategy(self, path='strategy'):
        dir_path = os.path.join(os.getcwd(), path)
        print(dir_path)
        # 遍历文件夹内所有.py文件
        for file_name in os.listdir(dir_path):
            # 判断是否为.py文件
            print(file_name)
            if os.path.splitext(file_name)[1] == '.py':
                # 动态加载模块
                module = os.path.splitext(file_name)[0]
                print(module)
                module = importlib.import_module("." + os.path.splitext(file_name)[0], path)
                g = {}
                # 调用init函数
                init_func = getattr(module, 'initialize', None)
                if init_func:
                    init_func(g)
                    self.context[file_name] = g

    def run_schedule(self):
        schedule.run_pending()
