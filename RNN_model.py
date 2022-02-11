"""
Minimal character-level Vanilla RNN model. Written by Andrej Karpathy (@karpathy)
BSD License
"""
import numpy as np
import sys

# data I/O
data = open("D:/input.txt", 'r').read()  # should be simple plain text file
#print(data)
chars = list(set(data))
#print(chars)  #중복없이 문자만 쏙쏙 ['f', 'u', 'l', 'n', 'p', '\n'..........]
data_size, vocab_size = len(data), len(chars)
print('data has %d characters, %d unique.' % (data_size, vocab_size))
char_to_ix = {ch: i for i, ch in enumerate(chars)}  #{'e': 0, 'h': 1, 'o': 2, 'l': 3} key= char, value=index
ix_to_char = {i: ch for i, ch in enumerate(chars)}  #{0: 'e', 1: 'h', 2: 'o', 3: 'l'} key=index, value= char
#print(char_to_ix)
#print(ix_to_char)

# hyperparameters
hidden_size = 100  # size of hidden layer of neurons
seq_length = 25  # number of steps to unroll the RNN for
learning_rate = 1e-1

# model parameters
Wxh = np.random.randn(hidden_size, vocab_size) * 0.01  # input to hidden/ x에서 RNN으로 들어로는 W
Whh = np.random.randn(hidden_size, hidden_size) * 0.01  # hidden to hidden/ hidden에서 오는 W
Why = np.random.randn(vocab_size, hidden_size) * 0.01  # hidden to output/ RNN cell에서 y로 넘어가는 W
bh = np.zeros((hidden_size, 1))  # hidden bias
by = np.zeros((vocab_size, 1))  # output bias


def lossFun(inputs, targets, hprev):
    """
    inputs,targets are both list of integers.
    hprev is Hx1 array of initial hidden state
    returns the loss, gradients on model parameters, and last hidden state
    """
    xs, hs, ys, ps = {}, {}, {}, {}
    hs[-1] = np.copy(hprev)  #hs = [~,HX1크기의 어레이]
    loss = 0  #초기 loss
    # forward pass
    #print(targets)
    #print(len(inputs))
    for t in range(len(inputs)):
        xs[t] = np.zeros((vocab_size, 1))  # encode in 1-of-k representation
        xs[t][inputs[t]] = 1
        hs[t] = np.tanh(np.dot(Wxh, xs[t]) + np.dot(Whh, hs[t - 1]) + bh)  # hidden state, hs[-1]= 최초 h_t-1가 저장된다. 그 바로 앞에 최종 h_t(hs[t])가 stack된다.
        ys[t] = np.dot(Why, hs[t]) + by  # unnormalized log probabilities for next chars // bias를 더한 형태로 y_t를 구함, y_t는 class score
        ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]))  # probabilities for next chars //다음 문자로 골라질 확룰을 나타낸 ps
        #print("targets{}: {}".format([targets[t], 0]))
        #print("ps{}: {}".format(t,ps[t][targets[t], 0]))
        loss += -np.log(ps[t][targets[t], 0])  # softmax (cross-entropy loss) input이 25보다 작으면 안됨....
    # backward pass: compute gradients going backwards
    #print("hs: {}".format(hs))
    dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
    dbh, dby = np.zeros_like(bh), np.zeros_like(by)
    dhnext = np.zeros_like(hs[0])
    ###################back propagation###############################
    for t in reversed(range(len(inputs))):  #거꾸로 가기 inputs = [0,1,2,3,4....] 해당 데이터의 인덱스들 t= inputs의 뒤에서부터 N,N-1,N-2......0
        dy = np.copy(ps[t]) #df= dy ps[t]
        #print("t is {}".format(t))
        #print("ps[t] is {}".format(ps[t]))
        dy[targets[t]] -= 1  # backprop into y. see http://cs231n.github.io/neural-networks-case-study/#grad if confused here 
        dWhy += np.dot(dy, hs[t].T)
        dby += dy
        dh = np.dot(Why.T, dy) + dhnext  # backprop into h
        dhraw = (1 - hs[t] * hs[t]) * dh  # backprop through tanh nonlinearity
        dbh += dhraw
        dWxh += np.dot(dhraw, xs[t].T)
        dWhh += np.dot(dhraw, hs[t - 1].T)
        dhnext = np.dot(Whh.T, dhraw)
    for dparam in [dWxh, dWhh, dWhy, dbh, dby]:  #gradient가 너무 커질 경우, clip. 너무 작아질 경우는 대비 못함(단점)-> RSTM
        np.clip(dparam, -5, 5, out=dparam)  # clip to mitigate exploding gradients
    return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs) - 1]


def sample(h, seed_ix, n):  #다음 번에 올 것으로 에측하는 문자의 인덱스가 담긴 리스트 ixes 반환
    """
    sample a sequence of integers from the model
    h is memory state, seed_ix is seed letter for first time step
    """
    x = np.zeros((vocab_size, 1))
    x[seed_ix] = 1
    ixes = []
    for t in range(n):  #200번 반복 200번 h 업데이트
        h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh)  #현재 상태의 h_t 구하기, h 업데이트, x는 현재 상태의 것 (inputs[0]의 인덱스만 1)
        y = np.dot(Why, h) + by  #class score y_t 구하기
        p = np.exp(y) / np.sum(np.exp(y))  #확률분포
        ix = np.random.choice(range(vocab_size), p=p.ravel())  #input.txt의 char 문자 개수 범위 안에서 p의 확률을 기반으로 랜덤으로 초이스 하나 하기, target chars의 index?
        x = np.zeros((vocab_size, 1))  #x 업데이트 , 반복문이 끝나면 최종적으로 예측하는 부분에만 1
        x[ix] = 1 #다음에 올 문자로 예측하는 것의 위치에 1
        ixes.append(ix)  #ixes에 다음에 올 문자로 예측하는 문자의 인덱스를 계속 추가
    #print(ixes)
    return ixes


n, p = 0, 0
mWxh, mWhh, mWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
mbh, mby = np.zeros_like(bh), np.zeros_like(by)  # memory variables for Adagrad
smooth_loss = -np.log(1.0 / vocab_size) * seq_length  # loss at iteration 0
while True:
    # prepare inputs (we're sweeping from left to right in steps seq_length long)
    if p + seq_length + 1 >= len(data) or n == 0:
        hprev = np.zeros((hidden_size, 1))  # reset RNN memory hidden_sizeX1
        p = 0  # go from start of data
    inputs = [char_to_ix[ch] for ch in data[p:p + seq_length]]  #[0,1,2,3,4] 제대로 된 해당 데이터의 인덱스들
    targets = [char_to_ix[ch] for ch in data[p + 1:p + seq_length + 1]]  #[1,2,3,4,5] inputs을 오른쪽으로 한 칸씩 미룬 배열
    #print(inputs)
    #print(ix_to_char[inputs[0]])
    # sample from the model now and then, p에 기반해 랜덤하게 골라진 예측값의 인덱스를 가지고 다시 문자로 반환, txt에는 다음에 올 것으로 예측하는 문자들
    if n % 100 == 0:
        sample_ix = sample(hprev, inputs[0], 200)
        txt = ''.join(ix_to_char[ix] for ix in sample_ix)
        print('----\n %s \n----' % (txt,))

    #loss를 줄여나가며 학습하는 부분#
    # forward seq_length characters through the net and fetch gradient
    loss, dWxh, dWhh, dWhy, dbh, dby, hprev = lossFun(inputs, targets, hprev)
    smooth_loss = smooth_loss * 0.999 + loss * 0.001
    if n % 100 == 0: print('iter %d, loss: %f' % (n, smooth_loss))  # print progress

    # perform parameter update with Adagrad
    for param, dparam, mem in zip([Wxh, Whh, Why, bh, by],
                                  [dWxh, dWhh, dWhy, dbh, dby],
                                  [mWxh, mWhh, mWhy, mbh, mby]):
        mem += dparam * dparam
        param += -learning_rate * dparam / np.sqrt(mem + 1e-8)  # adagrad update

    p += seq_length  # move data pointer
    n += 1  # iteration counter