import pretty_midi
import numpy as np
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

# Import MIDI file
midi_data = pretty_midi.PrettyMIDI('D:\\edge download\\Total Eclipse Of The Heart.mid')

# Extract tempo information
tempo_changes, tempos = midi_data.get_tempo_changes()
tempo_data = (tempo_changes, tempos)
# tempo_data is a tuple containing two arrays
# The first array is the time point (in seconds) at which tempo changes.
# The second array is the tempo value (in bpm) for the corresponding point in time

# Output tempo information
for time, tempo in zip(tempo_changes, tempos):
    print(f"At {time:.2f} seconds, the tempo is {tempo:.2f} BPM")

def analyze_tempo_data(tempo_data):
    # Unpack tempo_data to get the time change point and the corresponding BPM value respectively
    tempo_changes, tempos = tempo_data
    # Analyze the BPM area and calculate the area proportion
    # Computation time interval
    time_intervals = np.diff(tempo_changes)
    if len(time_intervals) < 1:  # If there is one tempo value
        return tempos[0]
    
    total_time = np.sum(time_intervals)
    time_ratio = time_intervals / total_time
    
    cumulative_ratio = 0
    for interval, tempo in zip(time_ratio, tempos[1:]):  # tempos[1:] Because the first point in time is 0, there's no interval
        cumulative_ratio += interval
        if cumulative_ratio > 0.5:
            return tempo  # Returns the specific BPM value for this region over 50%

    return tempos[-1]  # If not found, return the last BPM value

# Use this function to process tempo_data
dominant_bpm = analyze_tempo_data(tempo_data)
print(f"The BPM value that covers more than 50% of the time is: {dominant_bpm:.2f} BPM")

def classify_tempo(dominant_bpm, tempo_changes, tempos):
    # A result list, storing classified results
    tempo_types = []
    
    # Dictionary that defines the order of priorities (the smaller the number, the higher the priority)
    priority = {
        "Larghissimo": 1, "Lentissimo": 2, "Tranquillo": 3, 
        "Solenelle/Grave": 4, "Largo": 5, "Lentissimo": 6,
        "Larghetto": 7, "Adagio": 8, "Adagietto": 9,
        "Andante": 10, "Andantino": 11, "Marcia_moderato": 12,
        "Andante_moderato": 13, "Moderato": 14, "Allegretto": 15,
        "Allegro": 16, "Allegro_moderato": 17,  # Raise the priority of Allegro
        "Vivace": 18, "Vivacissimo/Allegrissimo": 19, "Presto": 20, "Prestissimo": 21
    }
    
    # Use a dictionary to record all possible types and corresponding averages
    possible_types = {}
    
    # Step 1: Categorized by dominant_bpm
    if dominant_bpm < 76:
        # Slow Tempo category
        possible_types["slow_tempo"] = None
        
        # Determine each rhythm type
        if all(bpm < 24 for bpm in tempos):
            possible_types["Larghissimo"] = np.mean(tempos)
        elif all(bpm < 48 for bpm in tempos):
            possible_types["Lentissimo"] = np.mean(tempos)
        elif np.mean([bpm for bpm in tempos if 73 <= bpm <= 83]) >= 80:
            possible_types["Tranquillo"] = np.mean([bpm for bpm in tempos if 73 <= bpm <= 83])
        
        # Further subdivision
        for bpm_range, label in [
            ([bpm for bpm in tempos if 25 <= bpm <= 45], "Solenelle/Grave"),
            ([bpm for bpm in tempos if 40 <= bpm <= 60], "Largo"),
            ([bpm for bpm in tempos if 45 <= bpm <= 60], "Lentissimo"),
            ([bpm for bpm in tempos if 60 <= bpm <= 66], "Larghetto"),
            ([bpm for bpm in tempos if 66 <= bpm <= 76], "Adagio"),
            ([bpm for bpm in tempos if 72 <= bpm <= 76], "Adagietto")
        ]:
            if bpm_range:
                possible_types[label] = np.mean(bpm_range)

    elif 76 <= dominant_bpm <= 120:
        # Moderate Tempo category
        possible_types["moderate_tempo"] = None
        
        # Determine each rhythm type
        for bpm_range, label in [
            ([bpm for bpm in tempos if 76 <= bpm <= 108], "Andante"),
            ([bpm for bpm in tempos if 80 <= bpm <= 108], "Andantino"),
            ([bpm for bpm in tempos if 83 <= bpm <= 85], "Marcia_moderato"),
            ([bpm for bpm in tempos if 92 <= bpm <= 112], "Andante_moderato"),
            ([bpm for bpm in tempos if 108 <= bpm <= 120], "Moderato"),
            ([bpm for bpm in tempos if 112 <= bpm <= 120], "Allegretto")
        ]:
            if bpm_range:
                possible_types[label] = np.mean(bpm_range)

    elif dominant_bpm > 120:
        # Fast Tempo category
        possible_types["fast_tempo"] = None
        
        # Determine each rhythm type
        for bpm_range, label in [
            ([bpm for bpm in tempos if 116 <= bpm <= 120], "Allegro_moderato"),
            ([bpm for bpm in tempos if 120 <= bpm <= 168], "Allegro"),
            ([bpm for bpm in tempos if 168 <= bpm <= 176], "Vivace"),
            ([bpm for bpm in tempos if 172 <= bpm <= 176], "Vivacissimo/Allegrissimo"),
            ([bpm for bpm in tempos if 168 <= bpm <= 200], "Presto"),
            ([bpm for bpm in tempos if bpm > 200], "Prestissimo")
        ]:
            if bpm_range:
                possible_types[label] = np.mean(bpm_range)
    
    # Sort and select the type with the highest priority
    sorted_types = sorted(
        possible_types.items(),
        key=lambda item: (priority.get(item[0], float('inf')), -item[1] if item[1] else 0)
    )

    # Select only the type with the highest priority
    if sorted_types:
        highest_priority_type = sorted_types[0][0]
        tempo_types.append(highest_priority_type)

    return tempo_types
tempo_classifications = classify_tempo(dominant_bpm, tempo_changes, tempos)
print("Tempo Classifications:", tempo_classifications)

def smooth_tempo_curve(tempos, tempo_types):
    # Set the upper and lower limits based on the type
    type_limits = {
        "Larghissimo": (None, 24 + 10),  # Upper limit only
        "Lentissimo": (None, 48 + 10),
        "Tranquillo": (None, 83 + 10),
        "Solenelle/Grave": (25 - 10, 45 + 10),
        "Largo": (40 - 10, 60 + 10),
        "Larghetto": (60 - 10, 66 + 10),
        "Adagio": (66 - 10, 76 + 10),
        "Adagietto": (72 - 10, 76 + 10),
        "Andante": (76 - 10, 108 + 10),
        "Andantino": (80 - 10, 108 + 10),
        "Marcia_moderato": (83 - 10, 85 + 10),
        "Andante_moderato": (92 - 10, 112 + 10),
        "Moderato": (108 - 10, 120 + 10),
        "Allegretto": (112 - 10, 120 + 10),
        "Allegro_moderato": (116 - 10, 120 + 10),
        "Allegro": (120 - 10, 168 + 10),
        "Vivace": (168 - 10, 176 + 10),
        "Vivacissimo/Allegrissimo": (172 - 10, 176 + 10),
        "Presto": (168 - 10, 200 + 10),
        "Prestissimo": (200 - 10, None)  # Lower limit only
    }
    
    # Select a primary type
    main_type = tempo_types[0]  # Use the first type in the classification as the primary type
    
    # Obtain upper and lower limits
    lower_limit, upper_limit = type_limits[main_type]
    
    # Adjust window_length dynamically to ensure that it is odd and does not exceed the data length
    window_length = min(len(tempos), 5)
    if window_length % 2 == 0:
        window_length -= 1  # Make sure it's odd number
    
    # The Savitzky-Golay filter is used to smooth the processing curve
    if window_length > 1:  # Ensure that window_length is valid
        smoothed_tempos = savgol_filter(tempos, window_length=window_length, polyorder=2)
    else:
        smoothed_tempos = tempos  # If window_length is invalid, the original data is returned
    
    # Apply upper and lower bound constraints
    if lower_limit is not None:
        smoothed_tempos = np.maximum(smoothed_tempos, lower_limit)
    if upper_limit is not None:
        smoothed_tempos = np.minimum(smoothed_tempos, upper_limit)
    
    return smoothed_tempos

# Use the above functions for smoothing
smoothed_tempos = smooth_tempo_curve(tempos=tempos, tempo_types=tempo_classifications)

# Draw the original and smoothed curves
plt.figure(figsize=(10, 6))
plt.plot(tempo_changes, tempos, label="Original Tempo", marker='o')
plt.plot(tempo_changes, smoothed_tempos, label="Smoothed Tempo", linestyle='--', marker='x')
plt.xlabel("Time (s)")
plt.ylabel("Tempo (BPM)")
plt.title("Original vs Smoothed Tempo")
plt.grid(True)
plt.legend()
plt.show()