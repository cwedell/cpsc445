# Assignment03

Colton Wedell\
CPSC 445-01\
10/30/2021

Sources used:\
https://stackoverflow.com/questions/3867890/count-character-occurrences-in-a-string-in-c \
https://www.geeksforgeeks.org/convert-string-char-array-cpp/ \
https://stackoverflow.com/questions/27890430/mpi-dynamically-allocation-arrays \
https://stackoverflow.com/questions/571394/how-to-find-out-if-an-item-is-present-in-a-stdvector \
https://stackoverflow.com/questions/17201590/c-convert-from-1-char-to-string \
https://mpitutorial.com/tutorials/mpi-scatter-gather-and-allgather/

Speedup calculation:
Assuming tB = time to run the baseline case with one processor, and tL = the time to traverse one link.

dna_count and dna_invert: 1 Bcast, 1 Scatter, 1 Gather\
Speedup = tB/(3(tL)(p-1))

dna_parse: 3 Bcast, 1 Scatter, 1 Gather\
Speedup = tB/(5(tL)(p-1))

Note: all of my programs use Bcast, Scatter, and Gather exclusively, meaning that the fastest topology is a star. If any of them had used Allgather or Alltoall, the fastest topology would be a completely connected one, in which case the (p-1) in each formula would be replaced with (p(p-1)/2), since a star network has p-1 links and a completely connected network has p(p-1)/2 links.
