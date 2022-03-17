import time
from collections import defaultdict

d = defaultdict(list)

def timeit(method):
    global d
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if ('log_time' in kw):
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = te - ts
            d[method.__name__].append(round(te - ts, 4))
        else:
            #print('%r  %2.3f s' % \
                  #(method.__name__, te - ts))
            d[method.__name__].append(round(te - ts, 4))
        return result
    return timed