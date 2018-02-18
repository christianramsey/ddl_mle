export PYTHONPATH=${PYTHONPATH}:${PWD}/trajectory

python -m train.task --batchsize=50 --traindir=../data/train --evaldir=../data/test --epochs=1 --outputdir=8 --hidden_units="23,42,13"