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
					for k in 0.9
					do
						for t in EDF
						do
							python src/simulator.py --test --iterate 1 -a $a -l $k -s $s --seed 1 --test_alg $t --episode $e -i 1.1 -x 3.0 #--job_distribution uniform
						done
					done
				done
			done
		done
	done
done
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
					for k in 0.9
					do
						for j in EDF
						do
							python src/simulator.py --train -a $a -s $s -l $k --seed $n --init_alg $j --episode $e --init_num $i --job_set_parameter ${e}_T${s}_${k}_s${n}_d1.1-3.0 -i 1.1 -x 3.0
							for t in proposal
							do
								python src/simulator.py --test -w model${e}_Nact${a}_T${s}_${k}_s${n}_init${j}${i} -a $a -s $s -l $k --seed $n --test_alg $t --episode $e --init_alg $j --init_num $i --iterate 1 --model_parameter ${e}_T${s}_${k}_s${n}_d1.1-3.0 -i 1.1 -x 3.0
							done
						done
					done
				done
			done
		done
	done
done
for n in 1
do
	for e in 10000 2000
	do
		for s in 500
		do
			for i in 0
			do
				for a in 5
				do
					for k in 0.9
					do
						for j in None
						do
							python src/simulator.py --train -a $a -s $s -l $k --seed $n --episode $e --init_alg $j --init_num $i --job_set_parameter ${e}_T${s}_${k}_s${n}_d1.1-3.0 -i 1.1 -x 3.0
							for t in proposal
							do
								python src/simulator.py --test -w model${e}_Nact${a}_T${s}_${k}_s${n}_init${j}${i} -a $a -s $s -l $k --seed $n --test_alg $t --episode $e --init_alg $j --init_num $i --iterate 1 --model_parameter ${e}_T${s}_${k}_s${n}_d1.1-3.0 -i 1.1 -x 3.0
							done
						done
					done
				done
			done
		done
	done
done
