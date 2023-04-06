
import time

import schedule

from controller.Task import Task


def main():
    t = Task()

    t.init_strategy()
    t.run_task()
    while True:
        # schedule.run_pending()
        t.run_schedule()
        time.sleep(1)


if __name__ == '__main__':
    main()
