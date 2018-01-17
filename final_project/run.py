import os

# Only use train
os.system("python main.py --name base --eval_mode 0 --model pytorch --skip_lstm 0 --batch_size 500 --init_type xavier")

os.system("python main.py --name lstm1 --eval_mode 0 --model pytorch --skip_lstm 1 --batch_size 500 --init_type xavier")
os.system("python main.py --name lstm2 --eval_mode 0 --model pytorch --skip_lstm 2 --batch_size 500 --init_type xavier")
os.system("python main.py --name lstm3 --eval_mode 0 --model pytorch --skip_lstm 3 --batch_size 500 --init_type xavier")

os.system("python main.py --name bt1b --eval_mode 0 --model pytorch --skip_lstm 0 --batch_size 100 --init_type xavier")
os.system("python main.py --name bt1k --eval_mode 0 --model pytorch --skip_lstm 0 --batch_size 1000 --init_type xavier")

os.system("python main.py --name mxnet --eval_mode 0 --model mxnet --skip_lstm 0 --batch_size 500 --init_type xavier")
os.system("python main.py --name tensorflow --eval_mode 0 --model tensorflow --skip_lstm 0 --batch_size 500 --init_type xavier")

os.system("python main.py --name kaiming --eval_mode 0 --model pytorch --skip_lstm 0 --batch_size 500 --init_type kaiming")
os.system("python main.py --name orthogonal --eval_mode 0 --model pytorch --skip_lstm 0 --batch_size 500 --init_type orthogonal")
os.system("python main.py --name normal --eval_mode 0 --model pytorch --skip_lstm 0 --batch_size 500 --init_type normal")

# use eval with best epoch
os.system("python main.py --epoch 42 --name base --eval_mode 1 --model pytorch --skip_lstm 0 --batch_size 500 --init_type xavier")

os.system("python main.py --epoch 36 --name lstm1 --eval_mode 1 --model pytorch --skip_lstm 1 --batch_size 500 --init_type xavier")
os.system("python main.py --epoch 29 --name lstm2 --eval_mode 1 --model pytorch --skip_lstm 2 --batch_size 500 --init_type xavier")
os.system("python main.py --epoch 26 --name lstm3 --eval_mode 1 --model pytorch --skip_lstm 3 --batch_size 500 --init_type xavier")

os.system("python main.py --epoch 9 --name bt1b --eval_mode 1 --model pytorch --skip_lstm 0 --batch_size 100 --init_type xavier")
os.system("python main.py --epoch 80 --name bt1k --eval_mode 1 --model pytorch --skip_lstm 0 --batch_size 1000 --init_type xavier")

os.system("python main.py --epoch 27 --name mxnet --eval_mode 1 --model mxnet --skip_lstm 0 --batch_size 500 --init_type xavier")
os.system("python main.py --epoch 28 --name tensorflow --eval_mode 1 --model tensorflow --skip_lstm 0 --batch_size 500 --init_type xavier")

os.system("python main.py --epoch 28 --name kaiming --eval_mode 1 --model pytorch --skip_lstm 0 --batch_size 500 --init_type kaiming")
os.system("python main.py --epoch 25 --name orthogonal --eval_mode 1 --model pytorch --skip_lstm 0 --batch_size 500 --init_type orthogonal")
os.system("python main.py --epoch 28 --name normal --eval_mode 1 --model pytorch --skip_lstm 0 --batch_size 500 --init_type normal")
