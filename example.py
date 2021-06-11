from timem import profile

if __name__ == "__main__":

    @profile(memory=False)
    def delay(secs):
        import time

        time.sleep(secs)

    @profile(timer=False)
    def isPal(s):
        return s == s[::-1]
    
    @profile
    def hello_world():
        print("hello world!")

    print(isPal("OOO"))
    print(isPal("OOOSS"))
    delay(3)
    hello_world()
