import sys

class Logger:
    msg = []
    prevmsgl = 0

    @classmethod
    def to_String(self, *m):
        b = ''
        for i in m:
            b += str(i)
        return b
    
    @classmethod
    def add_to_log(self, *m):
        Logger.msg.append(Logger.to_String(*m))

    @classmethod
    def print_log(self):
        for _ in range(Logger.prevmsgl):
            sys.stdout.write("\x1b[1A\x1b[2K")

        str_msg = ""
        for m in Logger.msg:
            str_msg += m + "\n"
        
        sys.stdout.write(str_msg)
        sys.stdout.flush()

        Logger.prevmsgl = len(Logger.msg)
        Logger.msg.clear()