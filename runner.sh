rm -r video_data
rm -r raft_results
rm -r performance
mkdir video_data
mkdir raft_results
mkdir performance

#python -u main.py --hypothesis 0 --method 'RAFT' --mode 'GOP0' --output 'results'
#python -u main.py --hypothesis 1 --method 'RAFT' --mode 'GOP0' --output 'results'


#python -u main.py --hypothesis 0 --method 'ICIP' --mode 'ALL' --output 'results'
#python -u main.py --hypothesis 1 --method 'ICIP' --mode 'ALL' --output 'results'
#python -u main.py --hypothesis 0 --method 'ICIP' --mode 'I0' --output 'results'
#python -u main.py --hypothesis 1 --method 'ICIP' --mode 'I0' --output 'results'
#python -u main.py --hypothesis 0 --method 'ICIP' --mode 'GOP0' --output 'results'
#python -u main.py --hypothesis 1 --method 'ICIP' --mode 'GOP0' --output 'results'

#python -u main.py --hypothesis 1 --method 'NEW' --mode 'ALL' --output 'results'
#python -u main.py --hypothesis 1 --method 'NEW' --mode 'I0' --output 'results'
#python -u main.py --hypothesis 1 --method 'NEW' --mode 'GOP0' --output 'results'

python -u main.py --hypothesis 0 --method 'RAFT' --mode 'ALL' --output 'performance'
python -u main.py --hypothesis 1 --method 'RAFT' --mode 'ALL' --output 'performance'
#python -u main.py --hypothesis 0 --method 'RAFT' --mode 'I0' --output 'results'
#python -u main.py --hypothesis 1 --method 'RAFT' --mode 'I0' --output 'results'

#python -u main.py --hypothesis 0 --method 'NEW' --mode 'ALL' --output 'results'
#python -u main.py --hypothesis 0 --method 'NEW' --mode 'I0' --output 'results'
#python -u main.py --hypothesis 0 --method 'NEW' --mode 'GOP0' --output 'results'

#python -u main.py --hypothesis 0 --method 'RND' --mode 'ALL' --output 'results'
#python -u main.py --hypothesis 1 --method 'RND' --mode 'ALL' --output 'results'
#python -u main.py --hypothesis 0 --method 'RND' --mode 'I0' --output 'results'
#python -u main.py --hypothesis 1 --method 'RND' --mode 'I0' --output 'results'
#python -u main.py --hypothesis 0 --method 'RND' --mode 'GOP0' --output 'results'
#python -u main.py --hypothesis 1 --method 'RND' --mode 'GOP0' --output 'results'

# python -u main.py --hypothesis 0 --method 'RND' --mode 'ALL' --videos 'vision/dataset' --fingerprint 'PRNU_fingerprints' --output 'results/' --gpu_dev /gpu:0  # >| output_H1.log &
