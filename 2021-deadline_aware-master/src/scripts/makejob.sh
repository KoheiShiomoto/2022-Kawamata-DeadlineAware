#! /bin/sh

#test data
for n in 1
do
	for e in 200
	do
		for s in 10000
		do
			for k in 0.9 0.8 0.7 0.6
			do
				python src/simulator.py --make --busy -s $s -l $k --seed $n --episode $e -i 1.1 -x 3.0 #--job_distribution pareto
            done
		done
	done
done

#training data
for n in 1
do
	for e in 2000
	do
		for s in 10 20 50 100 200 500 800 1000 1500 2000
		do
			for k in 0.9 0.8 0.7 0.6
			do
				python src/simulator.py --make --busy -s $s -l $k --seed $n --episode $e -i 1.1 -x 3.0 #--job_distribution pareto
            done
		done
	done
done
