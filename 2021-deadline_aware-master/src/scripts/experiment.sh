#! /bin/sh
for n in 1
do
	for e in 2000
	do
		for s in 10 20 50 100 200 500 800 1000 1500 2000
		do
			for i in 200
			do
				for a in 5
				do
					for k in 0.9 0.8 0.7 0.6
					do
						for j in EDF 
						do
							python src/simulator.py --train -a $a -s $s -l $k --seed $n --init_alg EDF --episode $e --init_alg $j --init_num $i --job_set_parameter 2000_T${s}_${k}_s${n}_d1.1-3.0 -i 1.1 -x 3.0
							for l in 0.9 0.8 0.7 0.6
							do
								for t in proposal
								do
									python src/simulator.py --test -w model${e}_Nact${a}_T${s}_${k}_s${n}_init${j}${i} --model_parameter 2000_T${s}_${k}_s${n}_d1.1-3.0 --iterate 200 -a $a -l $l -s 10000 --seed 1 --test_alg $t --episode $e --init_alg $j -i 1.1 -x 3.0
								done
							done
						done
					done
				done
			done
		done
	done
done
