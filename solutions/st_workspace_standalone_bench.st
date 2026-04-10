| size binaryString ones zeros s2 s1 t1 t2 |

size := 10000.

binaryString := ''.

size timesRepeat: [ binaryString := binaryString , (2 atRandom - 1) ].

t1 := Time millisecondsToRun: [ 50 timesRepeat: [s1 := binaryString sorted reverse join allButFirst , $1]].

t2 := Time millisecondsToRun: [ 50 timesRepeat: [
	ones := (binaryString count: [ :element | element = $1 ]) - 1.
	zeros := binaryString size - ones - 1.
	s2 := (String new: ones withAll: $1) , (String new: zeros withAll: $0) , $1
	]].

Transcript clear; show: 'Working....'; cr; cr;
	show: 'Same result is: '; show: s1 = s2; cr; cr;
	show: 'Sort Reverse: '; show: (t1 / 50) asFloat; show: 'ms.'; cr;
	show: 'Counting: '; show: (t2 / 50) asFloat; show: 'ms.'; cr; cr;
	show: 'After 50 iterations of '; show: size
