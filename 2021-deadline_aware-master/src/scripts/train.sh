#! /bin/sh
for n in 1
do
	for e in 10000
	do
		for s in 500
		do
			for i in 200
			do
				for a in 5
				do
					for k in 0.9 0.8 0.7 0.6
					do
						for j in EDF
						do
							python src/simulator.py --train -a $a -s $s -l $k --seed $n --init_alg EDF --episode $e --init_alg $j --init_num $i --job_set_parameter 10200_T${s}_${k}_s${n}_d1.1-3.0
						done
					done
				done
			done
		done
	done
done
