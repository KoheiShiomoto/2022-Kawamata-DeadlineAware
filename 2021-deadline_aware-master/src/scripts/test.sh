#! /bin/sh
for n in 1
do
	for e in 2000
	do
		for s in 10000
		do
			for i in 200
			do
				for a in 5
				do
					for k in 0.9 
					do
						for l in 0.9 0.8 0.7 0.6 
						do
							for t in EDF FIFO RR
							do
								python src/simulator.py --test --iterate 200 -a $a -l $l -s 10000 --seed 1 --test_alg $t --episode $e -i 1.1 -x 3.0 #--job_distribution pareto
							done
						done
					done
				done
			done
		done
	done
done
