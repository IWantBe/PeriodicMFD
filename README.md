# PeriodicMFD
The paper titled ”PeriodicMFD: A Periodic-based Framework for Multi-source Fault Diagnosis“ has been accepted for publication in IEEE Transactions on Transportation Electrification.

I would like to express my sincere gratitude to Mr. Zhang Tairui for his work on the code implementation, and also extend my heartfelt thanks to the other collaborators.

If you want to use this code, please
- install python and pytorch, and numpy, scipy, sklearn, etc.
- download cwru48k, jnu and hust 3 datasets, and put them into the [datasets](datasets) folder.
- run `python PeriodicMFD.py --dataset cwru48k --tasks [0,1,2] [0,1,3] [0,2,1] [0,2,3] [0,3,1] [0,3,2] [1,2,0] [1,2,3] [1,3,0] [1,3,2] [2,3,0] [2,3,1] --run 1` to run model in cwru48k with all tasks, `python PeriodicMFD.py --dataset jnu --tasks [600,800,1000] [600,1000,800] [800,1000,600] --run 1` in jnu, and `python PeriodicMFD.py --dataset hust --tasks [65,70,75] [65,70,80] [65,75,70] [65,75,80] [65,80,70] [65,80,75] [70,75,65] [70,75,80] [70,80,65] [70,80,75] [75,80,65] [75,80,70] --run 1` in hust.