import multiprocessing
import time

def func(msg):
    print "msg:", msg
    time.sleep(3)
    print "end"
    return "done" + msg

if __name__ == "__main__":
    # pool = multiprocessing.Pool(processes=6)
    # result = []
    # for i in xrange(3):
    #     msg = "hello %d" %(i)
    #     result.append(pool.apply_async(func, (msg, )))
    # pool.close()
    # pool.join()
    # for res in result:
    #     print ":::", res.get()
    # print "Sub-process(es) done."

    aList = [123, 123,'xyz', 'zara', 'abc', 'xyz'];

    aList.remove(123);
    print "List : ", aList;