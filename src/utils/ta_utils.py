import pandas as pd
import numpy as np
import warnings

def patched_psar(high, low, close, step=0.02, max_step=0.2):
    """
    Parabolic SAR implementation that avoids the FutureWarning from ta library.
    This is a replacement for ta.trend.psar_down that uses proper iloc indexing.
    
    Args:
        high: High prices
        low: Low prices
        close: Close prices
        step: Step parameter (default: 0.02)
        max_step: Maximum step parameter (default: 0.2)
        
    Returns:
        Pandas Series with PSAR values
    """
    # Suppress specific warning about Series.__setitem__
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning, 
                              message="Series.__setitem__ treating keys as positions is deprecated")
        
        # Get data length
        data_length = len(close)
        
        # Initialize output array
        psar = np.zeros(data_length)
        psar[:] = np.nan
        
        # Initialize variables
        ep = 0  # Extreme point
        af = step  # Acceleration factor
        trend = 1  # 1 for uptrend, -1 for downtrend
        
        # Set initial values
        psar[0] = low.iloc[0]
        ep = high.iloc[0]
        
        # Calculate PSAR values
        for i in range(1, data_length):
            # Previous PSAR value
            prev_psar = psar[i-1]
            
            # Calculate current PSAR
            if trend == 1:  # Uptrend
                psar[i] = prev_psar + af * (ep - prev_psar)
                # Ensure PSAR is below the previous two lows
                if i >= 2:
                    psar[i] = min(psar[i], low.iloc[i-1], low.iloc[i-2])
                    
                # Check for trend reversal
                if psar[i] > low.iloc[i]:
                    trend = -1
                    psar[i] = ep
                    ep = low.iloc[i]
                    af = step
                else:
                    # Update extreme point and acceleration factor
                    if high.iloc[i] > ep:
                        ep = high.iloc[i]
                        af = min(af + step, max_step)
            else:  # Downtrend
                psar[i] = prev_psar - af * (prev_psar - ep)
                # Ensure PSAR is above the previous two highs
                if i >= 2:
                    psar[i] = max(psar[i], high.iloc[i-1], high.iloc[i-2])
                    
                # Check for trend reversal
                if psar[i] < high.iloc[i]:
                    trend = 1
                    psar[i] = ep
                    ep = high.iloc[i]
                    af = step
                else:
                    # Update extreme point and acceleration factor
                    if low.iloc[i] < ep:
                        ep = low.iloc[i]
                        af = min(af + step, max_step)
        
        return pd.Series(psar, index=close.index)
