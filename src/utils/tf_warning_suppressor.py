"""
Utility to suppress common TensorFlow warnings.
Import this module at the beginning of main.py or other entry points to filter out
deprecated API warnings that come from third-party libraries.
"""
import warnings
import logging
import os

def suppress_tensorflow_warnings():
    """
    Suppress common TensorFlow deprecation warnings
    """
    # Set TensorFlow environment variable to suppress deprecation warnings globally
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=all, 1=info, 2=warning, 3=error
    
    # Filter specific TensorFlow warnings
    warnings.filterwarnings(
        "ignore", 
        message="The name tf.reset_default_graph is deprecated.*"
    )
    warnings.filterwarnings(
        "ignore", 
        message="From .*global_state.py.*"
    )
    warnings.filterwarnings(
        "ignore",
        category=DeprecationWarning
    )
    warnings.filterwarnings(
        "ignore",
        category=FutureWarning
    )
    
    # Suppress download messages from yfinance
    logging.getLogger('yfinance').setLevel(logging.ERROR)
    
    logging.info("Enhanced warning suppression applied for TensorFlow and other libraries")

# Suppress warnings when this module is imported
suppress_tensorflow_warnings()

# Import and configure TensorFlow with warning suppression
try:
    import tensorflow as tf
    # Disable eager execution warnings
    tf.get_logger().setLevel('ERROR')
    # Disable deprecation warnings
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
except ImportError:
    pass  # TensorFlow not available
