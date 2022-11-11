### Gradient-Descent neural network
Neural network with cross entropy lose function and sigma activation layer

#### the mathmatics implemented:
$$
C_i^k=-((1-y_i^k)ln(1-a_i^k)+y_i^klna_i^k)$$
\frac{\partial C}{\partial a_i^k}=
\frac{1-y_i^k}{1-a_i^k}-\frac{y_i^k}{a_i^k}=
\frac{a_i^k-y_i^ka_i^k-y_i^k+y_i^ka_i^k}{a_i^k(1-a_i^k)}=
\frac{a_i^k-y_i^k}{a_i^k(1-a_i^k)}
$$

$$
z_i^k=\sum_jw_{ij}^ka_j^{k-1}+b_i^k
$$

$$
a_i^k=\sigma(z_i^k)=\frac{1}{1+e^{-z_i^k}}
$$

$$
\frac{\partial a_i^k}{\partial z_i^k}=
\frac{e^{-z_i^k}}{(1+e^{-z_i^k})^2}=
\frac{1}{1+e^{-z_i^k}}-\frac{1}{(1+e^{-z_i^k})^2}
=a_i^k-(a_i^k)^2=a_i^k(1-a_i^k)
$$

$$
\delta_i^k=
\frac{\partial C}{\partial z_i^k}=
\frac{\partial C}{\partial a_i^k}\frac{\partial a_i^k}{\partial z_i^k}
=a_i^k-y_i^k (BP1)
$$

$$
\frac{\partial z_j^{k+1}}{\partial z_i^k}=
\frac{\partial }{\partial z_i^k}\sum_ow_{jo}^{k+1}
\sigma(z_o^k)+b_j^{k+1}=
w_{ji}^{k+1}a_i^k(1-a_i^k)
=w_{ji}^{k+1}\sigma'(z_i^k)
$$

$$
\delta_i^k=
\sum_j\frac{\partial C}{\partial z_j^{k+1}}
\frac{\partial z_j^{k+1}}{\partial z_i^k}
=\sum_j\delta_j^{k+1}w_{ji}^{k+1}\sigma'(z_i^k)
=\delta^{k+1}(w^{k+1})^T_i\sigma'(z_i^k)(BP2)
$$

$$
\frac{\partial z_j^k}{\partial w_{ji}^k}
=\frac{\partial }{\partial w_{ji}^k}\sum_ow_{jo}^ka_o^{k-1}+b_j^k
=a_i^{k-1}
$$

$$
\frac{\partial C}{\partial w_{ji}^k}=
\frac{\partial C}{\partial z_j^k}\frac{\partial z_j^k}{\partial w_{ji}^k}
=\delta_i^ka_i^{k-1}(BP3)
$$

$$
\frac{\partial C}{\partial b_j^k}=\delta_i^k(BP4)
$$

$$
\delta=a-y(BP1a)
$$

$$
\delta^k=\delta^{k+1}(w^{k+1})^T\sigma'(z^k)(BP2a)
$$

$$
\frac{\partial C}{\partial w_j^k}=\delta^ka^{k-1}(BP3a)
$$

$$
\frac{\partial C}{\partial b^k}=\delta^k(BP4a)
$$

#### using
```Python
import read_train_set as rs
import neural_network_cross_extropy as nc
l=rs.get()
A=nc.nn([28*28,30,10])
A.THG(l, 20, 1,330)
```
After 330 epoches, model could classify manual digit(in manual_digit folder) up to 99.9 percent.
![illusion_lose](illusion_lose.PNG)


Thank @mnielsen for [teaching script](http://neuralnetworksanddeeplearning.com/) <br>
תודה לאל keep us all the way
