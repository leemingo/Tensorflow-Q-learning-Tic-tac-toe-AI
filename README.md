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

* agent_RL : ��ȭ�н��� ���� �� ������Ʈ
* agent_Base : ����/�̱�� ���� �δ� �񱳿� ������Ʈ
* agent_Human : input�� �޾Ƽ� ���� ���� ������Ʈ

## Environment

* Tictactoe ���� ȯ���Դϴ�.
* step(action) : action�� �޾Ƽ� �����ϰ� observation�� ��ȯ�մϴ�.
* render() : ���� ���¸� ȭ�鿡 ����մϴ�.
* init()/reset() : ȯ���� �ʱ�ȭ�մϴ�.

## Learning Algorithm

* Neural network�� �̿��� Q-learning(off-policy TD control)�� ����Ͽ����ϴ�.
* Tensorflow�� ������ NN���� Optimal Action-value function�� �ٻ��ŵ�ϴ�.
* 1000 Episode�� �ڰ��������� Training�մϴ�.
* Experience replay������� sample���� ������ batch���� minibatch�� �̾� �н��մϴ�.
* 1000 Episode�� size�� 1000�� minibatch�� 20�� �н��մϴ�.
* 1000 Episode���� agent_Base�� 10000 Episode�� �׽�Ʈ�Ͽ� �Ҽ��� ���ڸ������� �·��� ����մϴ�.
* ���� ���� �·��� ���� ������ NN�� best model�� �����մϴ�.

* �н� ��

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

* agent_Base���� �������� �·��� 99%�� �����߽��ϴ�.
* ����� ����� ����(unbeatable)���� �н��� �Ǿ����ϴ�.
* DP-method�� Table�� �̿��� Q-learning�� ���� ������ �н��� �Ǿ����ϴ�.
* ��� state�� Table�� ������ �� �ִ� tictactoe���ӿ����� ȿ������ ����� �ƴմϴ�.
* state�� �ſ� ���ų�, ����� state�� ���ؼ� �����ϰ� �ൿ�ϴ°� ���� ������ ���ؼ��� ȿ������ �� �����ϴ�.

## Reference

* Richard S. Sutton and Andrew G. Barto. (2018). Reinforcement Learning : An Introduction. 
The MIT Press Cambridge, Massachusetts London, England