import tensorflow as tf
import numpy as np
from typing import Tuple

def env_step(env,action: np.ndarray) -> Tuple[np.ndarray,np.ndarray,np.ndarray]:
    state, reward, done, _ = env.step(action)
    return (
            state.astype(np.float32),
            np.array(reward,np.int32),
            np.array(done,np.int32)
        )
 
def tf_env_step(env,action: tf.Tensor) -> Tuple[tf.Tensor,tf.Tensor,tf.Tensor]:
    return tf.numpy_function(env_step,[env,action],[tf.float32,tf.int32,tf.int32])
