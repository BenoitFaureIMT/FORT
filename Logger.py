import sys

class Logger:
    #Logger
    msg = []
    prevmsgl = 0

    #Infos - flow
    mxFrameNumber = 0
    frameNumber = 0
    detectionTime = 0
    totalDetections = 0
    ReIDTimes = []
    totalTime = 0

    #Infos - end
    maxId = 0

    #Average times
    counter = 0
    detectionAvg = 0
    ReIDAvg = 0
    totalAvg = 0

    @classmethod
    def make_final_log(self):
        messages = ["", "Max ID", "Average Detection", "Average ReID", "Average FORT", "Average Total"]
        values = ["", Logger.maxId, Logger.detectionAvg / Logger.counter, Logger.ReIDAvg / Logger.counter, (Logger.totalAvg - Logger.detectionAvg) / Logger.counter, Logger.totalAvg / Logger.counter]
        units = ["", "", "ms", "ms", "ms", "ms"]

        mxLength = max([len(s) for s in messages])
        for i in range(len(messages)):
            Logger.add_to_log(messages[i], " " * (mxLength + 1 - len(messages[i])), values[i], " ", units[i])

    @classmethod
    def make_log(self):
        #Calculate message
        messages = ["Detections", "Detection", "", *["ReID" for i in range(len(Logger.ReIDTimes))], "", "FORT", "Total"]
        values = [Logger.totalDetections, int(Logger.detectionTime), "", *[int(v) for v in Logger.ReIDTimes], "", int(Logger.totalTime - Logger.detectionTime), int(Logger.totalTime)]
        units = ["", "ms", "", *["ms" for i in range(len(Logger.ReIDTimes))], "", "ms", "ms"]

        mxLength = max([len(s) for s in messages])
        hold = []
        for i in range(len(messages)):
            hold.append(Logger.to_String(messages[i], " " * (mxLength + 1 - len(messages[i])), values[i], " ", units[i]))

        #Log message
        mxLength = max([len(s) for s in hold]) - len(str(Logger.frameNumber)) - len(str(Logger.mxFrameNumber)) - 1
        Logger.add_to_log("")
        Logger.add_to_log("-" * (mxLength // 2), Logger.frameNumber, "/", Logger.mxFrameNumber, "-" * (mxLength - mxLength // 2))
        Logger.add_to_log("")

        for s in hold:
            Logger.add_to_log(s)

        #Update variables
        Logger.detectionAvg += Logger.detectionTime
        Logger.ReIDAvg += sum(Logger.ReIDTimes)
        Logger.totalAvg += Logger.totalTime

        Logger.ReIDTimes.clear()

        Logger.counter += 1


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

        # Logger.prevmsgl = len(Logger.msg)
        Logger.prevmsgl = len(str_msg.split("\n"))
        Logger.msg.clear()