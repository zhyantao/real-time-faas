import subprocess
import traceback
from time import time

tmp = '/tmp/'

"""
dd - convert and copy a file
man : http://man7.org/linux/man-pages/man1/dd.1.html

Options 
 - bs=BYTES
    read and write up to BYTES bytes at a time (default: 512);
    overrides ibs and obs
 - if=FILE
    read from FILE instead of stdin
 - of=FILE
    write to FILE instead of stdout
 - count=N
    copy only N input blocks
"""

cold = True


def main(args):
    global cold
    was_cold = cold
    cold = False
    try:
        start = time()
        bs = args.get("bs", 128)
        count = args.get("count", 1000)
        bs_cmd = 'bs=' + str(bs) + "k"
        count_cmd = 'count=' + str(count)

        out_fd = open(tmp + 'io_write_logs', 'w')
        dd = subprocess.Popen(['dd', 'if=/dev/zero', 'of=/tmp/out', bs_cmd, count_cmd], stderr=out_fd, stdout=out_fd)
        dd.communicate()
        latency = time() - start
        subprocess.check_output(['ls', '-alh', tmp])

        total_bytes = (bs * 1024) * count
        tput = "{0} bytes ({1} MBs), {2} s, {3} MB/s".format(total_bytes, total_bytes / (1024 * 1024), latency,
                                                             total_bytes / (1024 * 1024) / latency)
        with open(tmp + 'io_write_logs') as logs:
            result = logs.readlines()  # str(logs.readlines()[2]).replace('\n', '')
            return {"body": {"result": tput, "latency": latency, "cold": was_cold}}
    except Exception as e:
        return {"body": {"cust_error": traceback.format_exc(), "cold": was_cold}}
