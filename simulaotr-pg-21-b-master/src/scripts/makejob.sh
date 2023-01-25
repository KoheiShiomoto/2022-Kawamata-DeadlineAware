#! /bin/sh
for n in 1
do
	for e in 10000
	do
		for s in 500
		do
			for k in 0.9
			do
				python src/simulator.py --make -s $s -l $k --seed $n --episode $e -i 1.1 -x 3.0
            done
		done
	done
done
for n in 1
do
	for e in 2000
	do
		for s in 500
		do
			for k in 0.9
			do
				python src/simulator.py --make -s $s -l $k --seed $n --episode $e -i 1.1 -x 3.0
            done
		done
	done
done
for n in 1
do
	for e in 1
	do
		for s in 500
		do
			for k in 0.9
			do
				python src/simulator.py --make -s $s -l $k --seed $n --episode $e -i 1.1 -x 3.0
            done
		done
	done
done

