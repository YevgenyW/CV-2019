import numpy as np

class KF():
    def __init__(self, A, B, H, R, Q):
        self.A = A
        self.B = B
        self.H = np.atleast_2d(H)
        self.Q = Q
        self.P = np.eye(A.shape[0]) * 1000.
        self.x = np.zeros(A.shape[0])
        self.log_x = []
        self.xi = np.zeros(np.asarray(self.P.shape) + 1)
        if np.isscalar(R):
            self.Rinv = 1/R
        else:
            self.Rinv = np.linalg.inv(R)


    def predict(self, u=None):
        xminus = self.A.dot(self.x)
        if u is not None:
            xminus += self.B.dot(u)

        Pminus = self.A.dot(self.P).dot(self.A.T) + self.Q

        xi_vector = np.r_[xminus[np.newaxis], np.eye(self.x.shape[0])]

        self.xi = xi_vector.dot(np.linalg.inv(Pminus)).dot(xi_vector.T)
        self.x = xminus   # temporary
        self.P = Pminus   # temporary

        return xminus

    def update(self, y, Rinv=None):
        if Rinv is None:
            Rinv = self.Rinv
        y = np.atleast_2d(y).reshape(self.H.shape[0], -1)
        T_vector = np.concatenate((y, self.H), axis=1)
        T = T_vector.T.dot(Rinv).dot(T_vector)
        self.xi += T
        self.P = np.linalg.inv(self.xi[1:,1:])
        self.x = self.P.dot(self.xi[1:,0])

    def log(self):
        self.log_x.append(self.x.copy())