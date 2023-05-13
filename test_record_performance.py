import threading
import time
import psutil

def record_performance(stop_event):
    while not stop_event.is_set():
        cpu = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory().percent
        print("CPU : ", cpu)
        print("Memory : ", memory)
        performance['cpu'].append(cpu)
        performance['memory'].append(memory)

        time.sleep(1)


if __name__ == "__main__":
    event_stop = threading.Event()
    performance = {
        'cpu': [],
        'memory': [],
        'gpu': []
    }

    t3 = threading.Thread(target=record_performance, args=(event_stop,)).start()

    time.sleep(300)
    event_stop.set()
    import pdb;pdb.set_trace()