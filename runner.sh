#python -u main.py --hypothesis 1 --videos 'VISION/dataset' --fingerprint 'PRNU_fingerprints' --output 'results/results_h1_square' --gpu_dev /gpu:0  # >| output_H1.log &
python -u main.py --hypothesis 0 --videos 'VISION/dataset' --fingerprint 'PRNU_fingerprints' --output 'results_h0_IP' --gpu_dev /gpu:0  # >| output_H1.log &
