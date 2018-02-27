Tictactoe AI using Neural network Q-learning Method (Tensorflow)
=====================

## Run

### Training

```
$ python Train.py
```

### Play games

```
$ python Play.py
```

## Agents

* agent_RL : 강화학습을 진행 할 에이전트
* agent_Base : 랜덤/이기는 수를 두는 비교용 에이전트
* agent_Human : input을 받아서 수를 놓는 에이전트

## Environment

* Tictactoe 게임 환경입니다.
* step(action) : action을 받아서 실행하고 observation을 반환합니다.
* render() : 현재 상태를 화면에 출력합니다.
* init()/reset() : 환경을 초기화합니다.

## Learning Algorithm

* Neural network을 이용한 Q-learning(off-policy TD control)을 사용하였습니다.
* Tensorflow로 구현한 NN으로 Optimal Action-value function을 근사시킵니다.
* 1000 Episode의 자가대전으로 Training합니다.
* Experience replay방법으로 sample들을 저장한 batch에서 minibatch를 뽑아 학습합니다.
* 1000 Episode당 size가 1000인 minibatch로 20번 학습합니다.
* 1000 Episode마다 agent_Base와 10000 Episode씩 테스트하여 소숫점 네자리까지의 승률을 계산합니다.
* 가장 높은 승률이 나올 때마다 NN의 best model을 저장합니다.

* 학습 식

```
Q(S,A) = Q(S,A) + learning_rate * [R + discount_factor * Max(Q(S',a)) - Q(S,A)]

Y = R + discount_factor * Max(Q(S',a))

Loss(W) = MSE(Qw(S,A) - Y)

Gradient descent : W = W - learning_rate * gradient(Loss(W))
```

* Hyperparameter

```
1. learning rate(Adam_Optimizer) : 1e-2
2. discount_factor : 0.9
3. epsilon (egreedy method) : 0.1
```

## Conclusion

* agent_Base와의 대전에서 승률이 99%에 도달했습니다.
* 사람과 비슷한 수준(unbeatable)까지 학습이 되었습니다.
* DP-method나 Table을 이용한 Q-learning에 비해 느리게 학습이 되었습니다.
* 모든 state를 Table에 저장할 수 있는 tictactoe게임에서는 효율적인 방법은 아닙니다.
* state가 매우 많거나, 비슷한 state에 대해서 유사하게 행동하는게 옳은 문제에 대해서는 효과적일 것 같습니다.

## Reference

* Richard S. Sutton and Andrew G. Barto. (2018). Reinforcement Learning : An Introduction. 
The MIT Press Cambridge, Massachusetts London, England