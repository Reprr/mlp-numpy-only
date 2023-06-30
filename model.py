import numpy as np


class MLP:

    def __init__(self, in_size=10, out_size=3, hidden_szs=(100,)):
        self.in_size = in_size
        self.out_size = out_size
        self.hidden_sizes = hidden_szs
        self.params = self.init_params()
        self.layers = len(hidden_szs) + 2

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def softmax(selfs, x):
        sm = np.sum(np.exp(x), axis=-1)
        sm = sm[:, np.newaxis]
        return np.exp(x) / sm

    def init_params(self):
        params = {}
        input_dim = self.in_size
        for i, h_dim in enumerate(self.hidden_sizes):
            output_dim = h_dim
            weights_t = np.random.standard_normal((input_dim, output_dim))
            bias_t = np.random.standard_normal(output_dim)
            params['Weights_{}'.format(i)] = weights_t
            params['Bias_{}'.format(i)] = bias_t
            input_dim = output_dim
        output_dim = self.in_size
        weights_t = np.random.standard_normal((input_dim, output_dim))
        bias_t = np.random.standard_normal(output_dim)
        params['Weights_{}'.format(i + 1)] = weights_t
        params['Bias_{}'.format(i + 1)] = bias_t
        return params

    def forward_batch(self, batch_x):
        batch_size = batch_x.shape[0]
        out_lst = []
        out_lst.append(batch_x)
        for i in range(self.layers - 1):
            weights_t = self.params['Weights_{}'.format(i)]
            bias_t = self.params['Bias_{}'.format(i)]
            batch_x = np.dot(weights_t, batch_x) + bias_t
            if i != self.layers - 2:
                batch_x = self.sigmoid(batch_x)
                out_lst.append(batch_x)
            else:
                batch_x = self.softmax(batch_x)
                out_lst.append(batch_x)
        batch_y = batch_x
        return batch_y, out_lst

    def predict(self, x, batch_size=8):
        n, _ = x.shape
        output = np.zeros((n, self.out_size))
        for i in range(0, n, batch_size):
            batch_x = x[i: i + batch_size]
            batch_y, _ = self.forward_batch(batch_x)
            output[i: i + batch_size] = batch_y
        return output

    def backward_batch(self, batch_y_t, out_lst):
        params_g = {}
        # batch_size = batch_y_t.shape[0]
        for i in range(self.layers - 1, 0, -1):
            output = out_lst[i]
            h = out_lst[i - 1]
            if i == self.layers - 1:
                dL_ds = output - batch_y_t
                tmp = dL_ds
            else:
                dL_ds = output * (1 - output) * np.dot(tmp, self.params['W_{}'.format(i)].T)
                tmp = dL_ds
            ds_dw = h
            dL_dw = np.dot(ds_dw.T, dL_ds)
            dL_db = dL_ds.sum(axis=0)
            params_g['Weights_{}'.format(i - 1)] = dL_dw
            params_g['Bias_{}'.format(i - 1)] = dL_db
        return params_g

    def update_params(self, params_g, lr, v, momentum):
        for i, key in enumerate(params_g):
            v[key] = momentum * v[key] - lr * params_g[key]
            self.parameters[key] += v[key]
        return v

    def train(self, x, y, batch_size=8, epochs=20, lr=1e-2, momentum=0, shuffle=True):
        n = x.shape[0]
        inds = np.arange(n)
        for epoch in range(epochs):
            if shuffle:
                np.random.shuffle(inds)
            x = x[inds]
            y = y[inds]
            for i in range(0, n, batch_size):
                batch_x = x[i:i + batch_size]
                batch_y_t = y[i:i + batch_size]
                batch_y_pred, out_lst = self.forward_batch(batch_x)
                params_g = self.backward_batch(batch_y_t, out_lst)
                if epoch == 1 and i == 0:
                    v = {key: 0 for i, key in enumerate(params_g)}
                else:
                    v = self.update_params(params_g, lr, v, momentum)
                pred_y = self.predict(x)
                pred_y = np.argmax(pred_y, axis=1)
                true_y = np.argmax(y, axis=1)
                print('Train accuracy:{:.4f} \t'.format(np.mean(pred_y == true_y)))


