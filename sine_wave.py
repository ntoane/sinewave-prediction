# Imports
import numpy as np
import matplotlib.pyplot as plot


# Define a function to generate amplitude(y)
def gen_sine(time, frequency, amplitude, h_shift=0, v_shift=0) -> np.ndarray :
    return (amplitude * np.sin(2*np.pi * frequency * time + h_shift) + v_shift)

def main() :
    # Test the function
    sampling_rate = 10.0
    sample_interval = 1/sampling_rate
    time = np.arange(0, 10, sample_interval)

    frequency = 2
    amplitude = 1

    # Generate amplitude(y)
    amplitude_y = gen_sine(time, frequency, amplitude)

    print (f"Data set elements size: {len(amplitude_y)}")

    # Plot a sine wave using time and amplitude obtained from the sine wave function
    plot.figure(figsize = (16, 8))
    plot.plot(time, amplitude_y, 'b')
    # Give a title for the sine wave plot
    plot.title('Sine wave')
    # Give x axis label for the sine wave plot
    plot.xlabel('Time(s)')
    # Give y axis label for the sine wave plot
    plot.ylabel('Amplitude')
    plot.grid(True, which='both')
    plot.axhline(y=0, color='k')
    # Display the sine wave
    plot.show()

if __name__ == "__main__":
    main()