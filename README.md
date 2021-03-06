# Timem
## A simple decorator to profile memory usage and time taken by python functions

### Install
```pip install timem```

### Usage

In example.py:

```python
from timem import profile

@profile
def isPal(s):
    return s == s[::-1]

print(isPal("UwU"))
```
**Output:**
```
"isPal" took 10.012 ms to execute

Filename: example.py

Line #    Mem usage    Increment  Occurences   Line Contents
============================================================
    11     20.4 MiB     20.4 MiB           1       @profile
    12                                             def isPal(s):
    13     20.4 MiB      0.0 MiB           1           return s == s[::-1]


True
```

### Some configurations
*1. Only profile time*
```python
from timem import profile
@profile(memory=False)
def isPalindromeWithDelay(s):
    import time
    time.sleep(3)
    return s == s[::-1]

print(isPalindromeWithDelay("UwU"))
```
**Output:**
```
"isPal" took 10.028 ms to execute

True
```

*2. Only profile memory*
```python
from timem import profile

@profile(timer=False)
def isPalindrome(s):
    return s == s[::-1]

print(isPalindrome("sdasdf"))
```
**Output**
```
Filename: example.py

Line #    Mem usage    Increment  Occurences   Line Contents
============================================================
    11     20.6 MiB     20.6 MiB           1       @profile(timer=False)
    12                                             def isPal(s):
    13     20.6 MiB      0.0 MiB           1           return s == s[::-1]


True
```
