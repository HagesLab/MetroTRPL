import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model

class NeuralNetwork():
    """
    A neural network created from tensorflow/keras, which takes as input
    the parameters and/or initial conditions to, and outputs the simulated
    measurement from, a forward solver such as scipy.solve_ivp and odeint
    as defined in metropolis.model().
    """

    def __init__(self) -> None:
        self.model = keras.Model()
        self.model_scales = np.ndarray(shape=(0,), dtype=float)
        self.has_model = False

    def load_model(self, nn_fname : str, scales_fname : str) -> None:
        """
        Parameters
        ----------
        nn_fname : str
            Path to a keras weights/model .h5 file.
        scales_fname : str
            Path to an array of preprocessing scale factors.
        """
        self.model = load_model(nn_fname)
        self.model_scales = np.load(scales_fname, allow_pickle=True)
        self.has_model = True

    def preprocess(self, inputs : np.ndarray) -> None:
        """Scales the log of all input features to (-0.5, 0.5)"""
        inputs = np.log10(inputs)
        inputs -= self.model_scales[0]
        inputs /= self.model_scales[1]
        inputs -= 0.5

    def predict(self, t_steps : np.ndarray, inputs : np.ndarray) -> np.ndarray:
        """
        Predict TRPL measurement from given material parameters / initial excitation (inputs)
        at requested delay times (tSteps)
        """
        self.preprocess(inputs)

        coefs = self.model.predict(tf.constant(inputs))[0]

        self.postprocess(coefs)

        pl_from_nn = self.multiexp(t_steps, *coefs) # in [cm^-2 s^-1]
        return pl_from_nn

    def postprocess(self, outputs : np.ndarray) -> None:
        """
        Log of training outputs also scaled to (-0.5, 0.5), essentially
        Undo that here to get the actual outputs
        """
        outputs += 0.5
        outputs *= self.model_scales[3]
        outputs += self.model_scales[2]
        outputs[len(outputs)//2:] = 10 ** outputs[len(outputs)//2:]
        outputs[:len(outputs)//2] = -(10 ** outputs[:len(outputs)//2])


    def multiexp(self, x : np.ndarray, *args) -> np.ndarray:
        """
        Arbitrary-order multiexponential of form
        f(x) = a_0 * exp(k_0 * x) + a_1 * exp(k_1 * x) + ... + a_z * exp(k_z * x)
        
        in which args is a list of rates followed by coefs [k_0, k_1, ..., k_z, a_0, a_1, ..., a_z]

        Parameters
        ----------
        xin : 1D ndarray
            x values, e.g. delay time.
        *args : list-like
            Sequence of rates and coefs.

        Returns
        -------
        fit_y : 1D ndarray
            f(x) values.

        """
        fit_y = np.zeros_like(x, dtype=float)
        n_terms = len(args) // 2
        for i in range(n_terms):
            fit_y += args[i+n_terms] * np.exp(args[i] * x)

        return fit_y

if __name__ == "__main__":
    n = NeuralNetwork()